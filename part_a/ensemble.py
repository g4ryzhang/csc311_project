from neural_network import *
from torch.utils.data import RandomSampler


def load_data_rand_sample(base_path="../data"):
    """ Load the data in PyTorch Tensor. Training data is randomly sampled with
        replacement.
    :return: (zero_train_matrix_rsample, train_data_rsample, valid_data, test_data)
        WHERE:
        zero_train_matrix_rsample: 2D sparse matrix of randomly sampled training
        data where missing entries are filled with 0.
        train_data_rsample: 2D sparse matrix of randomly sampled training data
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    # Randomly sample from the training data with replacement
    train_matrix_rsample = RandomSampler(train_matrix, replacement=True, num_samples=train_matrix.shape[0])

    zero_train_matrix_rsample = train_matrix_rsample.copy()
    zero_train_matrix_rsample[np.isnan(train_matrix_rsample)] = 0

    zero_train_matrix_rsample = torch.FloatTensor(zero_train_matrix_rsample)
    train_matrix_rsample = torch.FloatTensor(train_matrix_rsample)

    return zero_train_matrix_rsample, train_matrix_rsample, valid_data, test_data


def ensemble_predict(model_lst, train_data, test_data):
    """ Average the predictions of all trained base models and evaluate.
    :param model_lst: list of Module
    :param train_data: 2D FloatTensor
    :param test_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """

    # Evaluate on all base models
    for model in model_lst:
        model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(test_data["user_id"]):

        pred = 0
        inputs = Variable(train_data[u]).unsqueeze(0)

        # Gather output of each models
        guesses = np.zeros(len(model_lst))

        for j in range(len(model_lst)):
            output = model_lst[j](inputs)
            guesses[j] = output[0][test_data["question_id"][i]].item() >= 0.5

            # Majority vote (Average the prediction)
        pred = np.sum(guesses) >= 2.0

        if pred == test_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix_rsample, train_matrix_rsample, valid_data, test_data = load_data()

    num_questions = train_matrix_rsample.shape[1]

    # hyperparameters.
    k = 10
    lr = 0.03
    num_epoch = 40
    lamb = 0.1

    model_lst = []  # Base models
    num_base_models = 3

    for i in range(num_base_models):
        model = AutoEncoder(num_questions, k)

        print("Training base model %d" % (i))

        train(model, lr, lamb, train_matrix_rsample, zero_train_matrix_rsample,
              valid_data, num_epoch)

        model_lst.append(model)

    valid_acc = ensemble_predict(model_lst, zero_train_matrix_rsample, valid_data)
    test_acc = ensemble_predict(model_lst, zero_train_matrix_rsample, test_data)
    print("Final validation accuracy = %f" % (valid_acc))
    print("Final test accuracy = %f" % (test_acc))


if __name__ == "__main__":
    main()