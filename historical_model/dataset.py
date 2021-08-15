class HistoricalModelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = df.values

    def __getitem__(self, idx):
        historical,

class HistoricalQueryDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=512,
            num_workers=16
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.test = None
        self.valid = None

        self.label_encoding = {}

    def prepare_data(self):
        logger.info('Reading Data')
        self.train = pd.read_pickle('datasets/aol_data_train_input_df.feather')
        self.valid = pd.read_pickle('datasets/aol_data_valid_input_df.feather')
        self.test = pd.read_pickle('datasets/aol_data_test_input_df.feather')

        self.label_encoding = {
            cat: i for i, cat in enumerate(
                self.valid['category'].unique()
            )
        }

    def train_dataloader(self):
        train_split = HistoricalModelDataset(self.train, self.label_encoding)
        return DataLoader(
            train_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        val_split = HistoricalModelDataset(self.valid, self.label_encoding)
        return DataLoader(
            val_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        test_split = HistoricalModelDataset(self.test, self.label_encoding)
        return DataLoader(
            test_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )