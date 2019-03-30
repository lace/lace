import os

try:
    # Baiji 2
    from baiji.pod import AssetCache, VersionedCache
    from baiji.pod.config import Config
    baiji_version_2 = True
except ImportError:
    # Baiji 3
    from baiji.pod.asset_cache import AssetCache
    from baiji.pod.versioned.versioned_cache import VersionedCache
    baiji_version_2 = False


MANIFEST_PATH_V2 = os.path.join(os.path.dirname(__file__), '..', 'manifest.json')
MANIFEST_PATH_V3 = os.path.join(os.path.dirname(__file__), '..', 'manifest_3.json')
IMMUTABLE_BUCKETS = ['bodylabs-versioned-assets', 'bodylabs-versioned-assets-tokyo']

if baiji_version_2:
    config = Config()
    config.CACHE_DIR = os.path.expanduser('~/.bodylabs_static_cache')
    config.IMMUTABLE_BUCKETS = IMMUTABLE_BUCKETS
    config.DEFAULT_BUCKET = os.getenv('ASSET_BUCKET', 'bodylabs-assets')

    sc = AssetCache(config)

    BUCKET = os.getenv('VC_BUCKET', 'bodylabs-versioned-assets')

    vc = VersionedCache(
        cache=sc,
        manifest_path=MANIFEST_PATH_V2,
        bucket=BUCKET)
else:
    sc = AssetCache(immutable_buckets=IMMUTABLE_BUCKETS)
    vc = VersionedCache(sc, MANIFEST_PATH_V3)
