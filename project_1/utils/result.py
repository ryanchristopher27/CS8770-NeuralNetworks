# Result Class

class Result():

    def __init__(
        self, 
        train_acc_vector, 
        val_acc_vector,
        train_loss_vector,
        val_loss_vector,
        y_pred,
        y_true,
        test_loss,
        test_accuracy,
        test_correct,
        hours,
        minutes,
        seconds,
    ):
        self.train_acc_vector = train_acc_vector
        self.val_acc_vector = val_acc_vector
        self.train_loss_vector = train_loss_vector
        self.val_loss_vector = val_loss_vector
        self.y_pred = y_pred
        self.y_true = y_true
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_correct = test_correct
        self.training_time = dict(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )

    # Getters
    def get_train_acc_vector(self):
        return self.train_acc_vector
    
    def get_val_acc_vector(self):
        return self.val_acc_vector
    
    def get_train_loss_vector(self):
        return self.train_loss_vector
    
    def get_val_loss_vector(self):
        return self.val_loss_vector
    
    def get_y_pred(self):
        return self.y_pred
    
    def get_y_true(self):
        return self.y_true
    
    def get_test_loss(self):
        return self.test_loss
    
    def get_test_accuracy(self):
        return self.test_accuracy
    
    def get_test_correct(self):
        return self.test_correct
    
    def get_training_time(self):
        return self.training_time
    
