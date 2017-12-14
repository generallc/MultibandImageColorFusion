import argparse
import models_residual


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--mode', type=str, default='train', help="Choose train or eval")
    parser.add_argument('--data_file', type=str, default="../../data/processed/ColorFusion_128_data.h5", help="Path to HDF5 containing the data")
    parser.add_argument('--training_mode', default="in_memory", type=str,
                        help=('Training mode. Choose in_memory to load all the data in memory and train.'
                              'Choose on_demand to load batches from disk at each step'))
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--n_batch_per_epoch', default=50, type=int, help="Number of batches per epoch")
    parser.add_argument('--nb_epoch', default=80, type=int, help="Number of training epochs")
    parser.add_argument('--nb_resblocks', default=2, type=int, help="Number of residual blocks for simple model")
    parser.add_argument('--nb_neighbors', default=10, type=int, help="Number of nearest neighbors for soft encoding")
    parser.add_argument('--epoch', default=5, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--T', default=0.2, type=float,
                        help=("Temperature to change color balance in evaluation phase."
                              "If T = 1: desaturated. If T~0 vivid"))

    parser.add_argument('--img_dim', default=(3, 128, 128), type=tuple, help="set the training image size")
    parser.add_argument('--sub_name', default='', type=str, help="set the sub name of the model")
    parser.add_argument('--model_name', default='Colorization', type=str, help="set the model name of the model, only use in test_class.py")

    parser.add_argument('--history', default='1.txt', type=str, help="the output dictionary of the loss")
    args = parser.parse_args()

    # Set default params


    d_params0 = {"data_file": args.data_file,
                 "batch_size": args.batch_size,
                 "n_batch_per_epoch": args.n_batch_per_epoch,
                 "nb_epoch": args.nb_epoch,
                 "nb_resblocks": args.nb_resblocks,
                 "training_mode": args.training_mode,
                 "nb_neighbors": args.nb_neighbors,
                 "epoch": args.epoch,
                 "T": 0.5,
                 "sub_name": "t0.5",
                 "img_dim": args.img_dim,
                 "model_name": 'ResidualImageColorizationModel',
                 "history": 'Residual_up.txt',
                 "lr": 0.001,
                 }


    d_params1 = {"data_file": args.data_file,
                 "batch_size": args.batch_size,
                 "n_batch_per_epoch": args.n_batch_per_epoch,
                 "nb_epoch": args.nb_epoch,
                 "nb_resblocks": args.nb_resblocks,
                 "training_mode": args.training_mode,
                 "nb_neighbors": args.nb_neighbors,
                 "epoch": args.epoch,
                 "T": 0.5,
                 "sub_name": "t0.5",
                 "img_dim": args.img_dim,
                 "model_name": 'ResidualImageColorizationModel_trans',
                 "history": 'Residual_trans.txt',
                 "lr": 0.001,
                 }


    d_params2 = {"data_file": args.data_file,
                 "batch_size": args.batch_size,
                 "n_batch_per_epoch": args.n_batch_per_epoch,
                 "nb_epoch": args.nb_epoch,
                 "nb_resblocks": args.nb_resblocks,
                 "training_mode": args.training_mode,
                 "nb_neighbors": args.nb_neighbors,
                 "epoch": args.epoch,
                 "T": 0.5,
                 "sub_name": "t0.5",
                 "img_dim": args.img_dim,
                 "model_name": 'ResidualImageColorizationModel_subpixel',
                 "history": 'Residual_subpixel.txt',
                 "lr": 0.001,
                 }




    Flag = True
    if Flag:
        params = [d_params0]

        for x in params:
            # Launch training

            """
            Plot the models
            """

            model = models_residual.ResidualImageColorizationModel().create_model(**x)

            """
            Train Colorization
            """
            colorization = models_residual.ResidualImageColorizationModel()

            colorization.create_model(**x)

            model = colorization.fit(**x)

            """
            Colorization prediction
            """
            # predict_main(**x)

    Flag1 = True
    if Flag1:
        params = [d_params1]

        for x in params:
            # Launch training

            """
            Plot the models
            """

            model = models_residual.ResidualImageColorizationModel_trans().create_model(**x)

            """
            Train Colorization
            """
            colorization = models_residual.ResidualImageColorizationModel_trans()

            colorization.create_model(**x)

            model = colorization.fit(**x)

            """
            Colorization prediction
            """
            # predict_main(**x)


    Flag2 = True
    if Flag2:
        params = [d_params2]

        for x in params:
            # Launch training

            """
            Plot the models
            """

            model = models_residual.ResidualImageColorizationModel_subpixel().create_model(**x)

            """
            Train Colorization
            """
            colorization = models_residual.ResidualImageColorizationModel_subpixel()

            colorization.create_model(**x)

            model = colorization.fit(**x)

            """
            Colorization prediction
            """
            # predict_main(**x)

