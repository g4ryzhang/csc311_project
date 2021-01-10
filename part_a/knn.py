from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy (impute by question): %f" % (acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    
    param_k = [1, 6, 11, 16, 21, 26]
    num_k = len(param_k)
    k_acc = np.zeros(num_k)

    # by student similarity
    for i in range(num_k):
        k_acc[i] = knn_impute_by_user(sparse_matrix, val_data, param_k[i])

    k_star = param_k[np.argmax(k_acc)]
    k_star_test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)

    print("Imputed by student similarity")
    print("k = %f has the best performance, with test accuracy %f" % 
        (int(k_star), k_star_test_acc))


    # by question similarity
    # transpose so that distance is calculated by question
    # s_mat_T = sparse_matrix.T   
    # for i in range(num_k):
    #     k_acc[i] = knn_impute_by_item(s_mat_T, val_data, param_k[i])

    # k_star = param_k[np.argmax(k_acc)]
    # k_star_test_acc = knn_impute_by_item(s_mat_T, test_data, k_star)

    # print("Imputed by question similarity")
    # print("k = %f has the best performance, with test accuracy %f" % 
    #     (int(k_star), k_star_test_acc))


    # Plot and report
    fig, ax = plt.subplots()
    ax.plot(param_k, k_acc)
    ax.set_xticks(param_k)
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('accuracy')
    fig.show()
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
