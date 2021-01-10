from utils import *
from autograd import grad
import autograd.numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} transformed to a matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.

    for i in range(len(data["user_id"])):
        if data["is_correct"][i] == 1:
            log_lklihood += np.log(sigmoid(theta[data["user_id"][i]] 
                            - beta[data["question_id"][i]]))
        else:
            log_lklihood += np.log(1 - sigmoid(theta[data["user_id"][i]] 
                            - beta[data["question_id"][i]]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    d_theta = grad(neg_log_likelihood, 1)   # derivative w.r.t theta
    d_beta = grad(neg_log_likelihood, 2)    # derivative w.r.t beta

    # "Alternating" gradient descent instead of simultaneous
    theta -= d_theta(data, theta, beta) * lr
    beta -= d_beta(data, theta, beta) * lr

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_nlld_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.ones(542)
    beta = np.ones(1774)

    val_acc_lst = []
    train_nlld_lst = []     # negative log likelihood

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_nlld_lst.append(neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nlld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    n_iter = 15

    theta_train, beta_train, val_acc_lst, train_nlld_lst = irt(train_data, 
                                                        val_data, lr, n_iter)
    


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    
    test_acc = evaluate(test_data, theta_train, beta_train)
    print("The validation set accuracy is %f" % (val_acc_lst[-1]))
    print("The test accuracy of the trained model is %f" % (test_acc))
    
    # Plot and report
    _, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(0, n_iter), train_nlld_lst)
    ax[0].set_xlabel('# iterations')
    ax[0].set_ylabel('training negative log likelihood')
    ax[1].plot(np.arange(0, n_iter), val_acc_lst)
    # ax[1].set_xticks(np.arange(0, n_iter+1))
    ax[1].set_xlabel('# iterations')
    ax[1].set_ylabel('validation accuracy')
    plt.show()


    #####################################################################
    # Part d
    #####################################################################

    # Randomly select 5 questions
    np.random.seed(0)
    questions = np.random.randint(1774, size=5)
    prob_lst = np.zeros((5, 542))   # output probabilities

    # Vectorize sigmoid function for one question over all students
    vec_sig = np.vectorize(sigmoid)

    _, ax = plt.subplots() # plot

    for j in range(5):
        beta_j = beta_train[questions[j]]
        prob_lst[j] = vec_sig(theta_train - beta_j)
        ax.plot(theta_train, prob_lst[j], 'x', label='question %d' % (questions[j]))

    ax.legend()
    ax.set_xlabel('theta')
    ax.set_ylabel('probability of correct response')
    plt.show()



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
