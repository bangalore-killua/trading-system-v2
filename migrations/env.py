import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

load_dotenv()

# Import all ORM models so autogenerate can see them
from database import (  # noqa: F401
    Base,
    Signal,
    Position,
    Trade,
    ScreenerResult,
    EquityCurve,
    WatchlistHistory,
    ScoreHistory,
    SystemEvent,
    PerformanceSummary,
    TradeFeatures,
    OptionPosition,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Read DB URL from environment — never from alembic.ini
def get_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        # Fallback: build from individual env vars (same logic as database.py)
        from urllib.parse import quote_plus
        host     = os.getenv("DB_HOST",     "localhost")
        port     = os.getenv("DB_PORT",     "5432")
        dbname   = os.getenv("DB_NAME",     "trading_system")
        user     = os.getenv("DB_USER",     "killua_trading")
        password = quote_plus(os.getenv("DB_PASSWORD", ""))
        url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
