from sklearn.model_selection import train_test_split

from openadmet.models.split.split_base import SplitterBase, splitters


@splitters.register("ShuffleSplitter")
class ShuffleSplitter(SplitterBase):
    """
    Vanilla splitter, uses sklearn's train_test_split which wraps ShuffleSplit
    """

    def split(self, X, y, y_err=None):
        """
        Split the data into train, validation, and test sets

        Parameters
        ----------
        X : Iterable
            Features to split
        y : Iterable
            Targets to split
        y_err : Optional[Iterable]
            Optional errors for targets to split
        Returns
        -------
        tuple
            A tuple containing the train, validation, and test sets for features and targets.
        """

        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                train_size=None,
                test_size=int(self.val_size * X.shape[0]),
                random_state=self.random_state,
            )
            return X_train, X_val, None, y_train, y_val, None

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            y_err if y_err is not None else None,
            train_size=None,
            test_size=int(self.test_size * X.shape[0]),
            random_state=self.random_state,

        )

        # No validation set requested, return train(+val) and test sets
        if self.val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test

        # Split train+val into train and val sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test
