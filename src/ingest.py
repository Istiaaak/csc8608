from data.data import MVTecDataset

def ingest(
    cls: str,
    size: int = 224,
    vanilla: bool = True,
    out_path: str = "./datasets"
) -> str:

    ds = MVTecDataset(cls=cls, size=size, vanilla=vanilla)
    # À ce stade, ds.train_ds.root.parent vaut "./datasets/bottle"
    return str(ds.train_ds.root.parent)
