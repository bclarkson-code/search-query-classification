class HistoricalQueryDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=512,
            num_workers=16
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_df = None
        self.test_df = None
        self.valid_df = None

        self.train = None
        self.test = None
        self.valid = None

        self.label_encoding = {}

    def prepare_data(self):
        logger.info('Reading Data')
        self.train = pd.read_feather('aol_data_train.feather')
        self.test = pd.read_feather('aol_data_test.feather')
        self.valid = pd.read_feather('aol_data_valid.feather')

        logger.info('Preparing Inputs')
        self.train = build_inputs(self.train_df)
        self.test = build_inputs(self.test_df)
        self.valid = build_inputs(self.valid_df)

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