#!/bin/sh
# Build the index on first start unless the image already carries one, then
# serve. MERCATORIA_FORCE_REBUILD=1 forces a rebuild on an existing volume.
set -eu

cd /app

if [ "${MERCATORIA_FORCE_REBUILD:-0}" = "1" ] || [ ! -f data/index/current.txt ]; then
    echo "building index..."
    python -m mercatoria --config config.json rebuild
fi

exec python -m mercatoria --config config.json serve \
    --host 0.0.0.0 \
    --port "${PORT:-8080}"
