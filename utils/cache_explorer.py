import diskcache

if __name__ == '__main__':
    cache = diskcache.Cache('../calibration/cache')

    print(cache._sql('SELECT key FROM Cache').fetchall())