#!/usr/bin/env python3

class SVDD():
    def __init__(self, params):
        self.params =params 
        

    # self.support_vectors
    # self.kernel
    # self.center
    # self.radius

    def train(path_to_data):
        """
        train the model
        """
        pass

    def save(path_to_model):
        """
        save the model to "path_to_model"
        """
        pass

    def load(path_to_model):
        """
        load the model to "path_to_model"
        """
        pass


    def dist_from_center_to(x):
        """
        compute dist from center to x in RKHS
        """
        pass


class Decoder_with_SVDD(SVDD):
    def map(z):
        """
        [0, 1]^n -> X,
        where X is a feasible domain defined as {x | g_j(x) <= 0, for all j}
        and g_j: R^n -> R is a constraint function.
        """
        pass


def main():
    path_to_data = "path/to/data.npy" # (n, m)-array
    path_to_model = "path/to/model"
    svdd_params = {"blah", "blah"}

    # construct SVDD model
    svdd = SVDD(svdd_params)

    # train SVDD model
    svdd.train(path_to_data)

    # save model
    svdd.save(path_to_model)

    # load trained SVDD
    model = svdd.load(path_to_model)

    decoder = Decoder_with_SVDD(model)

    dim_z = 10
    z = np.random.uniform(low=0., high=1., size=dim_z)

    # map [0, 1]^n to X
    x = decoder.map(z)


if __name__ == "__main__":
    main()
