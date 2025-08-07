class DrugResponseDataPreparer:
    def create_synthetic_data(self):
        # Return dummy synthetic data
        return {"data": "synthetic"}

    def prepare_training_data(self, synthetic_data):
        # Return dummy prepared data
        return {
            "X_train": [[0, 1], [1, 0]],
            "y_train": [0, 1]
        }