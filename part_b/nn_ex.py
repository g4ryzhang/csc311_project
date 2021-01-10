from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from matplotlib import pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.
    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, first, second):
        """ Initialize a class AutoEncoder.
        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # encorders
        self.enc1 = nn.Linear(num_question, first)
        self.enc2 = nn.Linear(first, second)
        # self.enc3 = nn.Linear(64, 16)
        # decorders
        # self.dec1 = nn.Linear(16, 64)
        self.dec1 = nn.Linear(second, first)
        self.dec2 = nn.Linear(first, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.
        :return: float
        """
        enc1_w_norm = torch.norm(self.enc1.weight, 2)
        enc2_w_norm = torch.norm(self.enc2.weight, 2)
        # enc3_w_norm = torch.norm(self.enc3.weight, 2)
        dec1_w_norm = torch.norm(self.dec1.weight, 2)
        dec2_w_norm = torch.norm(self.dec2.weight, 2)
        # dec3_w_norm = torch.norm(self.dec3.weight, 2)
        return enc1_w_norm + enc2_w_norm + dec1_w_norm + dec2_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.
        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        out = torch.sigmoid(self.enc1(inputs))
        out = torch.sigmoid(self.enc2(out))
        # out = torch.sigmoid(self.enc3(out))
        out = torch.sigmoid(self.dec1(out))
        out = torch.sigmoid(self.dec2(out))
        # out = torch.sigmoid(self.dec3(out))

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.
    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    # num_question = train_data.shape[1]

    # Training and validation data for plot
    loss_lst = []
    val_acc_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)

            # Regularized loss
            loss = loss + (0.5 * lamb * model.get_weight_norm())

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        loss_lst.append(train_loss)
        val_acc_lst.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        with open("q3c_log", "a") as file:
            file.write("Epoch: {} \tTraining Cost: {:.6f}\t "
                       "Valid Acc: {}\n".format(epoch, train_loss, valid_acc))

    # Plot training/validation data

    plt.plot(np.arange(num_epoch), val_acc_lst, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    report = "Validation Accuracy: {}%".format(round(val_acc_lst[-1]*100, 5))
    plt.text(18, 0.65, report)
    plt.savefig("{}-{}".format(model.enc2.in_features, model.enc2.out_features))
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.
    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    num_questions = train_matrix.shape[1]

    # Set model hyperparameters.
    model = AutoEncoder(num_questions, first=512, second=10)

    # Set optimization hyperparameters.
    lr = 0.03
    num_epoch = 45
    lamb = 0.01

    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Test accuracy = %f" % (test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

