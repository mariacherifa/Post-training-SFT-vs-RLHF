######################################################################
#                          Advantage function 
######################################################################
import torch 

def cumulative_future_returns(token_rewards, gamma = 1): 

    B,T = token_rewards.shape
    
    returns = torch.zeros_like(token_rewards)

    running_return = torch.zeros_like(token_rewards[:, 0])

    for i in range(T-1, -1, -1):
        running_return = token_rewards[:, i] + gamma * running_return
        returns[:,i] = running_return
    
    return returns

def advantage(value_model, full_ids, token_rewards, gamma=1):

    returns = cumulative_future_returns(token_rewards, gamma=gamma)

    # Get all predicted values
    all_values = value_model(full_ids)

    # Number of prompt tokens
    P = full_ids.shape[1] - token_rewards.shape[1]

    # Select values before each generated action
    values = all_values[:, P - 1:-1]

    advantages = returns - values.detach()

    return advantages

