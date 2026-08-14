from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar, Literal

from polars._utils.logging import eprint
from polars._utils.wrap import wrap_ldf
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    _convert_iceberg_to_object_store_storage_options,
)
from polars.io.iceberg._utils import _normalize_windows_iceberg_file_uri
from polars.io.partition import _InternalPlPathProviderConfig

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import gen_uuid_v7

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyiceberg.catalog
    import pyiceberg.table

    import polars as pl
    from polars._plr import PyLazyFrame
    from polars._typing import EngineType, StorageOptionsDict


def _partition_key_exprs(table: pyiceberg.table.Table) -> list[pl.Expr] | None:
    spec = table.spec()

    if not spec.fields:
        return None

    from pyiceberg.transforms import (
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )
    from pyiceberg.types import BinaryType, IntegerType, LongType, StringType

    import polars as pl

    schema = table.schema()
    reserved_names = {field.name for field in schema.fields}
    exprs: list[pl.Expr] = []

    for field in spec.fields:
        source_field = schema.find_field(field.source_id)
        source_type = source_field.field_type
        transform = field.transform
        expr = pl.col(source_field.name)

        if isinstance(transform, IdentityTransform):
            pass
        elif isinstance(
            transform, (YearTransform, MonthTransform, DayTransform, HourTransform)
        ):
            if type(source_type).__name__ in {
                "TimestamptzType",
                "TimestamptzNanoType",
            }:
                expr = expr.dt.convert_time_zone("UTC")

            if isinstance(transform, YearTransform):
                expr = expr.dt.year() - 1970
            elif isinstance(transform, MonthTransform):
                expr = (expr.dt.year() - 1970) * 12 + expr.dt.month() - 1
            elif isinstance(transform, DayTransform):
                expr = expr.cast(pl.Date).cast(pl.Int32)
            else:
                expr = expr.dt.epoch("us") // 3_600_000_000
        elif isinstance(transform, TruncateTransform):
            if isinstance(source_type, (IntegerType, LongType)):
                expr = expr - expr % transform.width
            elif isinstance(source_type, StringType):
                expr = expr.str.slice(0, transform.width)
            elif isinstance(source_type, BinaryType):
                expr = expr.bin.slice(0, transform.width)
            else:
                msg = (
                    "sink to Iceberg table with "
                    f"'{transform}' partition transform on '{source_type}'"
                )
                raise NotImplementedError(msg)
        else:
            msg = f"sink to Iceberg table with '{transform}' partition transform"
            raise NotImplementedError(msg)

        key_name = f"__POLARS_ICEBERG_PARTITION_{field.field_id}"
        while key_name in reserved_names:
            key_name += "_"
        reserved_names.add(key_name)
        exprs.append(expr.alias(key_name))

    return exprs


@dataclass(kw_only=True)
class IcebergSinkState:
    py_catalog_class_module: str
    py_catalog_class_qualname: str

    catalog_name: str
    catalog_properties: dict[str, str]

    table_name: str
    mode: Literal["append", "overwrite", "upsert", "delete", "overwrite_files"]
    snapshot_properties: dict[str, str]
    iceberg_storage_properties: StorageOptionsDict

    base_snapshot_id: int | None
    data_file_paths_to_delete: list[str]

    sink_uuid_str: str

    table_: NoPickleOption[pyiceberg.table.Table]
    commit_result_df: NoPickleOption[pl.DataFrame]

    @staticmethod
    def new(
        target: str | pyiceberg.table.Table,
        *,
        mode: Literal[
            "append", "overwrite", "upsert", "delete", "overwrite_files"
        ] = "append",
        snapshot_properties: dict[str, str] | None = None,
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig | None = None,
        storage_options: StorageOptionsDict | None = None,
    ) -> IcebergSinkState:
        catalog_config = (
            (
                IcebergCatalogConfig._from_api_parameter_or_environment_default(
                    catalog,
                    fn_name="sink_iceberg",
                )
            )
            if isinstance(target, str)
            else (
                IcebergCatalogConfig(
                    class_=type(target.catalog),
                    name=target.catalog.name,
                    properties=target.catalog.properties,
                )
            )
        )

        from pyiceberg.catalog.noop import NoopCatalog

        if catalog_config.class_ is NoopCatalog:
            msg = (
                "cannot sink to static Iceberg table: "
                f"{type(target) = }, {getattr(target, 'catalog', None) = }"
            )
            raise TypeError(msg)

        return IcebergSinkState(
            py_catalog_class_module=catalog_config.class_.__module__,
            py_catalog_class_qualname=catalog_config.class_.__qualname__,
            catalog_name=catalog_config.name,
            catalog_properties=catalog_config.properties,
            table_name=target if isinstance(target, str) else ".".join(target.name()),
            mode=mode,
            snapshot_properties=snapshot_properties or {},
            iceberg_storage_properties=storage_options or {},
            base_snapshot_id=None,
            data_file_paths_to_delete=[],
            sink_uuid_str=gen_uuid_v7().hex(),
            table_=NoPickleOption(target if not isinstance(target, str) else None),
            commit_result_df=NoPickleOption(),
        )

    def table(self) -> pyiceberg.table.Table:
        if self.table_.get() is None:
            module = importlib.import_module(self.py_catalog_class_module)
            qualname_split = self.py_catalog_class_qualname.split(".")

            catalog_class: type[pyiceberg.catalog.Catalog] = getattr(
                module, qualname_split[0]
            )

            for part in qualname_split[1:]:
                catalog_class = getattr(catalog_class, part)

            catalog = catalog_class(self.catalog_name, **self.catalog_properties)
            self.table_.set(catalog.load_table(self.table_name))

        return self.table_.get()  # type: ignore[return-value]

    def _get_converted_storage_options(self) -> dict[str, str]:
        return _convert_iceberg_to_object_store_storage_options(
            self.iceberg_storage_properties
        )

    def attach_sink(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return wrap_ldf(lf._ldf.sink_iceberg(self))

    def copy_on_write(
        self,
        lf: pl.LazyFrame,
        *,
        on: str | Sequence[str],
        engine: EngineType,
    ) -> pl.DataFrame:
        import polars as pl

        keys = [on] if isinstance(on, str) else list(on)
        if not keys or len(keys) != len(set(keys)):
            msg = "`on` must contain at least one unique column name"
            raise ValueError(msg)

        table = self.table()
        source_schema = lf.collect_schema()
        missing_source_keys = set(keys).difference(source_schema)
        if missing_source_keys:
            msg = (
                "copy-on-write keys not found in source schema: "
                f"{sorted(missing_source_keys)}"
            )
            raise ValueError(msg)

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        table_schema = pl.Schema(schema_to_pyarrow(table.schema()))
        missing_table_keys = set(keys).difference(table_schema)
        if missing_table_keys:
            msg = (
                "copy-on-write keys not found in Iceberg schema: "
                f"{sorted(missing_table_keys)}"
            )
            raise ValueError(msg)

        if self.mode == "upsert" and source_schema != table_schema:
            msg = "upsert source schema must match the Iceberg table schema"
            raise pl.exceptions.SchemaError(msg)

        source_keys = lf.select(keys).collect(engine=engine)
        if source_keys.is_empty():
            return self._set_commit_result(table.metadata_location)

        if any(source_keys.null_count().row(0)):
            msg = "copy-on-write keys cannot contain null values"
            raise ValueError(msg)

        unique_source_keys = source_keys.unique()
        if self.mode == "upsert" and unique_source_keys.height != source_keys.height:
            msg = "upsert source contains duplicate keys"
            raise pl.exceptions.DuplicateError(msg)

        snapshot = table.current_snapshot()
        if snapshot is None:
            if self.mode == "delete":
                return self._set_commit_result(table.metadata_location)

            self.mode = "append"
            self.attach_sink(lf).collect(engine=engine)
            return self.commit_result_df.get()  # type: ignore[return-value]

        file_path_column = "__POLARS_ICEBERG_COW_FILE_PATH"
        while file_path_column in table_schema:
            file_path_column += "_"

        from polars.io.iceberg.functions import scan_iceberg

        target = scan_iceberg(
            table,
            snapshot_id=snapshot.snapshot_id,
            reader_override="native",
            _include_file_paths=file_path_column,
        )
        touched_paths = (
            target.select(*keys, file_path_column)
            .join(unique_source_keys.lazy(), on=keys, how="semi")
            .select(file_path_column)
            .unique()
            .collect(engine=engine)
            .get_column(file_path_column)
            .to_list()
        )

        if not touched_paths:
            if self.mode == "delete":
                return self._set_commit_result(table.metadata_location)

            self.mode = "append"
            self.attach_sink(lf).collect(engine=engine)
            return self.commit_result_df.get()  # type: ignore[return-value]

        retained_rows = (
            target.filter(pl.col(file_path_column).is_in(touched_paths))
            .drop(file_path_column)
            .join(unique_source_keys.lazy(), on=keys, how="anti")
        )
        output = (
            pl.concat([retained_rows, lf], how="vertical")
            if self.mode == "upsert"
            else retained_rows
        )

        self.mode = "overwrite_files"
        self.base_snapshot_id = snapshot.snapshot_id
        self.data_file_paths_to_delete = touched_paths
        self.attach_sink(output).collect(engine=engine)

        return self.commit_result_df.get()  # type: ignore[return-value]

    def _attach_resolved_sink(self, plf: PyLazyFrame) -> PyLazyFrame:
        from pyiceberg.table import TableProperties
        from pyiceberg.utils.properties import property_as_bool, property_as_int

        import polars as pl

        table = self.table()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        partition_key_exprs = _partition_key_exprs(table)

        if table.sort_order().fields:
            msg = "sink to Iceberg table with sort order"
            raise NotImplementedError(msg)

        if location_provider_impl := table_properties.get(
            TableProperties.WRITE_PY_LOCATION_PROVIDER_IMPL
        ):
            msg = (
                "sink to Iceberg table with custom location provider"
                f" '{location_provider_impl}'"
            )
            raise NotImplementedError(msg)

        object_storage_enabled = property_as_bool(
            table_properties,
            TableProperties.OBJECT_STORE_ENABLED,
            TableProperties.OBJECT_STORE_ENABLED_DEFAULT,
        )
        object_storage_partitioned_paths = (
            property_as_bool(
                table_properties,
                TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS,
                TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS_DEFAULT,
            )
            if object_storage_enabled
            else None
        )

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        arrow_schema = schema_to_pyarrow(table.schema())

        approximate_bytes_per_file = 2 * 1024 * 1024 * 1024

        if v := property_as_int(
            properties=table_metadata.properties,
            property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        ):
            estimated_compression_ratio = 4
            approximate_bytes_per_file = min(
                estimated_compression_ratio * v, (1 << 64) - 1
            )

        return (
            wrap_ldf(plf)
            .sink_parquet(
                pl.PartitionBy(
                    _normalize_windows_iceberg_file_uri(
                        self.sink_base_path(
                            object_storage_enabled=object_storage_enabled
                        )
                    ),
                    file_path_provider=PlIcebergPathProviderConfig(
                        object_storage_partitioned_paths=object_storage_partitioned_paths
                    ),
                    key=partition_key_exprs,
                    include_key=False if partition_key_exprs is not None else None,
                    approximate_bytes_per_file=approximate_bytes_per_file,
                ),
                arrow_schema=arrow_schema,
                storage_options=self._get_converted_storage_options(),
                lazy=True,
            )
            ._ldf
        )

    def commit(self, data_file_paths: list[str]) -> pl.DataFrame:
        import polars._utils.logging

        function_start_instant = perf_counter()
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(f"IcebergSinkState[commit]: mode: '{self.mode}'")

        table = self.table()

        original_metadata_location = table.metadata_location

        if sys.platform == "win32":
            data_file_paths = [
                (f"file://{p[8:]}" if p.startswith("file:///") else p)
                for p in data_file_paths
            ]

        with table.transaction() as tx:
            if self.mode == "overwrite":
                from pyiceberg.expressions import AlwaysTrue

                tx.delete(AlwaysTrue(), snapshot_properties=self.snapshot_properties)

            start_instant = perf_counter()

            if self.mode == "overwrite_files":
                self._overwrite_files(tx, data_file_paths)
            else:
                if verbose:
                    eprint("IcebergSinkState[commit]: begin add_files")

                start_instant = perf_counter()

                tx.add_files(
                    data_file_paths,
                    snapshot_properties=self.snapshot_properties,
                    check_duplicate_files=False,
                )

            if verbose:
                elapsed = perf_counter() - start_instant
                eprint(f"IcebergSinkState[commit]: finish add_files ({elapsed:.3f}s)")
                eprint("IcebergSinkState[commit]: begin transaction commit")

            start_instant = perf_counter()

        if verbose:
            now = perf_counter()
            elapsed = now - start_instant
            eprint(
                f"IcebergSinkState[commit]: finish transaction commit ({elapsed:.3f}s)"
            )
        else:
            now = None

        new_metadata_location = table.metadata_location

        assert new_metadata_location != original_metadata_location

        self._set_commit_result(new_metadata_location)

        if now is not None:
            total_elapsed = now - function_start_instant

            eprint(
                f"IcebergSinkState[commit]: finished, total elapsed time: {total_elapsed:.3f}s"
            )

        return self.commit_result_df.get()  # type: ignore[return-value]

    def _overwrite_files(
        self,
        transaction: pyiceberg.table.Transaction,
        data_file_paths: list[str],
    ) -> None:
        from pyiceberg.io.pyarrow import parquet_files_to_data_files
        from pyiceberg.table import TableProperties

        table = self.table()
        assert self.base_snapshot_id is not None

        current_snapshot = table.current_snapshot()
        if (
            current_snapshot is None
            or current_snapshot.snapshot_id != self.base_snapshot_id
        ):
            msg = "Iceberg table changed while planning copy-on-write operation"
            raise RuntimeError(msg)

        files_by_path = {
            _normalize_windows_iceberg_file_uri(task.file.file_path): task.file
            for task in table.scan(snapshot_id=self.base_snapshot_id).plan_files()
        }
        missing_paths = set(self.data_file_paths_to_delete).difference(files_by_path)
        if missing_paths:
            msg = f"copy-on-write data files not found in base snapshot: {sorted(missing_paths)}"
            raise RuntimeError(msg)

        if transaction.table_metadata.name_mapping() is None:
            transaction.set_properties(
                {
                    TableProperties.DEFAULT_NAME_MAPPING: transaction.table_metadata.schema().name_mapping.model_dump_json()
                }
            )

        added_files = parquet_files_to_data_files(
            io=table.io,
            table_metadata=transaction.table_metadata,
            file_paths=iter(data_file_paths),
        )
        with transaction.update_snapshot(
            snapshot_properties=self.snapshot_properties
        ).overwrite() as overwrite:
            for path in self.data_file_paths_to_delete:
                overwrite.delete_data_file(files_by_path[path])
            for data_file in added_files:
                overwrite.append_data_file(data_file)

    def _set_commit_result(self, metadata_location: str) -> pl.DataFrame:
        import polars as pl

        result = pl.DataFrame(
            {"metadata_path": metadata_location},
            schema={"metadata_path": pl.String},
            height=1,
        )
        self.commit_result_df.set(result)
        return result

    def sink_base_path(self, *, object_storage_enabled: bool) -> str:
        from pyiceberg.table import TableProperties

        table = self.table()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        sink_base_path = (
            path.rstrip("/")
            if (path := table_properties.get(TableProperties.WRITE_DATA_PATH))
            else f"{table_metadata.location.rstrip('/')}/data"
        )

        if object_storage_enabled:
            return f"{sink_base_path}/"

        return f"{sink_base_path}/{self.sink_uuid_str}/"


@dataclass(frozen=True, kw_only=True)
class PlIcebergPathProviderConfig(_InternalPlPathProviderConfig):
    pl_path_provider_id: ClassVar[str] = "iceberg"
    extension: ClassVar[Literal["parquet"]] = "parquet"
    object_storage_partitioned_paths: bool | None = None
