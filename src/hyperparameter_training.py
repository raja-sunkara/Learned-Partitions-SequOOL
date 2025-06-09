# hyperparameter tuning imports
import torch
import torch.nn as nn
import ray
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
#from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from functools import partial
from src.utils.nn import one_layer_net
from tqdm import tqdm
from src.utils.general import set_seed


def train_mlp(config, X, Y):
    
    # Model setup
    mlp_model = one_layer_net(X.shape[1], config['NUM_HIDDEN'], 1)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['STEP_SIZE'], gamma=config['GAMMA'])
    
    # Full training
    pbar = tqdm(range(5000), desc = f"Training")
    for epoch in pbar:

        optimizer.zero_grad()

        yhat = mlp_model(X)
        loss = torch.nn.functional.mse_loss(yhat,Y,reduction='mean')
        loss.backward()
        optimizer.step()

        if epoch %10 ==0:
            pbar.set_postfix({'Training Loss': loss.item()})
        if scheduler is not None:
                scheduler.step()


    final_loss = torch.nn.functional.mse_loss(mlp_model(X),Y,reduction='mean').item()     # final loss
    
    train.report({"loss":final_loss})



def neural_network_training(X, Y, cfg,logger,torch_seed,num_samples=10, gpus_per_trial=0):
    
    set_seed(torch_seed)
    hyperparameter_space = {
        'NUM_HIDDEN': tune.choice(cfg.hyperparameter.NUM_HIDDEN['values']),
        'LEARNING_RATE': tune.loguniform(cfg.hyperparameter.LEARNING_RATE.lower, 
                                         cfg.hyperparameter.LEARNING_RATE.upper),
        'WEIGHT_DECAY': tune.loguniform(cfg.hyperparameter.WEIGHT_DECAY.lower, 
                                        cfg.hyperparameter.WEIGHT_DECAY.upper),
        'GAMMA': tune.uniform(cfg.hyperparameter.GAMMA.lower, 
                              cfg.hyperparameter.GAMMA.upper),
        'STEP_SIZE': tune.choice(cfg.hyperparameter.STEP_SIZE['values']),
    }

    scheduler = ASHAScheduler( metric="loss", mode="min" )   # this scheduler has a early stopping
    
    result = tune.run(
        partial(train_mlp, X=X, Y=Y),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=hyperparameter_space,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final training loss: {best_trial.last_result['loss']}")


    # Now you can train your model with the best hyperparameters
    def train_final_model(X, Y, hyperparameters):
        model = one_layer_net(X.shape[1], hyperparameters['NUM_HIDDEN'], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['LEARNING_RATE'], weight_decay=hyperparameters['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters['STEP_SIZE'], gamma=hyperparameters['GAMMA'])
        
        for _ in tqdm(range(5000), desc="Training Final Model"):
            optimizer.zero_grad()
            yhat = model(X)
            loss = nn.functional.mse_loss(yhat, Y, reduction='mean')
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        return model

    best_trained_model = train_final_model(X, Y, best_trial.config)


    return best_trained_model


