import chromadb, inspect
print('chromadb version:', getattr(chromadb, '__version__', 'unknown'))
print('Client dir:', [n for n in dir(chromadb.Client) if not n.startswith('_')])
print('Client doc:', chromadb.Client.__doc__)
try:
    import chromadb.config as cconf
    print('Config keys sample:', list(cconf.__dict__.keys())[:20])
except Exception as e:
    print('config import error', e)
