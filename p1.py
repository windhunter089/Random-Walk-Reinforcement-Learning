import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Create function to generate sequence
#seq is sequence in list
#sec_vector is sequence vector
def sequence_gen(neighbors, state_vector):
    seq = []
    reward = 0
    current_state = "D"
    seq.append(current_state)
    seq_vector = state_vector[current_state]

    #Iterate to generate sequence
    while current_state != "A" and current_state != "G":
        next_state = neighbors[current_state][random.randint(0,1)]
        seq.append(next_state)
        if next_state != "A" and next_state != "G":
            seq_vector = np.concatenate((seq_vector, state_vector[next_state]), axis = 1)
            # print(seq_vector)
        #terminal state G
        elif next_state == "G":
            reward = 1
        #update current state for next loop
        current_state = next_state
    
    # print("seq:", seq)
    # print("seqvec:", curr_state_vector)
    # print("r:", reward)
    
    return (seq,seq_vector,reward)

def plot_f3(rmse):
    rmse_df = pd.DataFrame(list(rmse.items()), columns = ["λ", "Error"])
    rmse_df[["Error"]] = rmse_df[["Error"]] / no_set
    # print(rmse_df)

    rmse_df.plot(x="λ", y="Error", linestyle='-', marker='o', legend=False)
    plt.xlim(rmse_df["λ"].min() - 0.05, rmse_df["λ"].max() + 0.05)
    plt.ylim(rmse_df["Error"].min() - 0.03, rmse_df["Error"].max() + 0.03)
    plt.ylabel("Error")
    plt.margins(x=0.5, y=0.15)
    plt.annotate("Widrow-Hoff", xy=(1.0, 0.8), xytext=(0.76, 0.75))
    # plt.show()
    plt.savefig('figure_3.png')
    plt.clf()

def plot_f4(rmse2):
    rmse2_df = pd.DataFrame({'α':alpha_list})
    # print("RMSE2 ",rmse2_df)
    for lamda in [0,0.3,0.8,1]:
        rmse2_temp = pd.DataFrame(list(rmse2[lamda].items()), columns = ["α", "λ = "+str(lamda)])
        rmse2_temp[["λ = "+str(lamda)]] = rmse2_temp[["λ = "+str(lamda)]] / no_set
        rmse2_df = rmse2_df.merge(rmse2_temp, on = "α", how = "left")
    # print(rmse2_df)
    
    #Plot
    plt.plot("α", "λ = 0", data = rmse2_df, marker = 'o')
    plt.plot("α", "λ = 0.3", data = rmse2_df, marker = 'o')
    plt.plot("α", "λ = 0.8", data = rmse2_df, marker = 'o')
    plt.plot("α", "λ = 1", data = rmse2_df, marker = 'o')
    plt.legend()
    plt.margins(x=0.15, y=0.15)
    plt.xlim(-0.05, 0.65)
    plt.ylim(0.05, 0.6)
    plt.ylabel("rmse")
    plt.xlabel("α")
    # plt.show()
    plt.savefig('figure_4.png')
    plt.clf()

def plot_f5(rmse2):
    rmse2_df = pd.DataFrame({'α':alpha_list})
    # print("RMSE2 ",rmse2_df)
    for lamda in lambda_list:
        rmse2_temp = pd.DataFrame(list(rmse2[lamda].items()), columns = ["α", "λ = "+str(lamda)])
        rmse2_temp[["λ = "+str(lamda)]] = rmse2_temp[["λ = "+str(lamda)]] / no_set
        rmse2_df = rmse2_df.merge(rmse2_temp, on = "α", how = "left")
    # print(rmse2_df)
    rmse2_df_5 = pd.DataFrame(rmse2_df.iloc[:,1:].min()).reset_index()
    rmse2_df_5.columns = ['λ','Error using best alpha']
    rmse2_df_5['λ'] = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # print(rmse2_df_5)
    #Plot
    rmse2_df_5.plot("λ","Error using best alpha", marker ='o',legend = False)
    plt.margins(x=0.2, y=0.05)
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.07, 0.19)
    plt.ylabel("Error using best α")
    plt.annotate("Widrow-Hoff", xy=(1.0, 0.18), xytext=(0.76, 0.18))
    # plt.show()
    plt.savefig('figure_5.png')
    



if __name__ == "__main__":

    #Calculate the ideal weight.
    #Base on equation 5 from the paper [(I - Q)^-1] h
    # I = np.identity(5)
    #Qij is tranition matrix from i to j
    i_q_matrix = np.identity(5) - np.array([
        [0,0.5,0,0,0],
        [0.5,0,0.5,0,0],
        [0,0.5,0,0.5,0],
        [0,0,0.5,0,0.5],
        [0,0,0,0.5,0]
    ])
    # print(i_q_matrix)
    ideal_weight = np.dot(np.linalg.inv(i_q_matrix), np.array([0,0,0,0,0.5]))
    ideal_weight = ideal_weight.reshape(5,1)
    # print(ideal_weight)

    # Set up environment for MDP, represent transition dynamic
    neighbors = {
        "A": ["B"],
        "B": ["C","A"],
        "C": ["B","D"],
        "D": ["C","E"],
        "E": ["D","F"],
        "F": ["E","G"],
        "G": ["F"]
    }
    #State vector to represent each state according to the paper
    state_vector = {
        "B": np.array([[1,0,0,0,0]]).transpose(),
        "C": np.array([[0,1,0,0,0]]).transpose(),
        "D": np.array([[0,0,1,0,0]]).transpose(),
        "E": np.array([[0,0,0,1,0]]).transpose(),
        "F": np.array([[0,0,0,0,1]]).transpose()
    }
    # print(state_vector)

    #Generating training data set , 100 training set with 10 sequence in each set
    no_set = 100
    no_seq = 10

    #Init for training data set
    seq = [[] for _ in range(no_set)]
    training_data = [[] for _ in range(no_set)]
    reward = [[] for _ in range(no_set)]

    random.seed(328)
    np.random.seed(328)
    #Populate the structure using sequence generation function
    for i in range(no_set):
        for j in range(no_seq):
            s,sv,r = sequence_gen(neighbors,state_vector)
            seq[i].append(s)
            training_data[i].append(sv)
            reward[i].append(r)
    # print("seq:", seq)
    # print("training data:", training_data)
    # print("r:", reward)
            
    #lambda list for the experiment
    lambda_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #init data for the error on lamda
    rmse = {
        0: 0.0,
        0.1: 0.0,
        0.2: 0.0,
        0.3: 0.0,
        0.4: 0.0,
        0.5: 0.0,
        0.6: 0.0,
        0.7: 0.0,
        0.8: 0.0,
        0.9: 0.0,
        1.0: 0.0
    }
    # print(rmse)

    #Figure 3: Calculate rmse for TD lamda using different lamda value
    #epsilon for converse threshold
    epsilon = 0.01
    alpha = 0.2
    for lamda in lambda_list:
        for i in range(no_set):
            #init for weight 
            weight = np.array([[0.5,0.5,0.5,0.5,0.5]]).transpose()

            #iterate until converge
            while True:
                previous_weight = weight
                # set up accumulate delta weight
                delta_weight_acc = np.array([[0.0,0.0,0.0,0.0,0.0]]).transpose()
                for j in range(no_seq):
                    err = np.array([[0.0,0.0,0.0,0.0,0.0]]).transpose()
                    delta_weight = np.array([[0.0,0.0,0.0,0.0,0.0]]).transpose()
                    for k in range((training_data[i][j]).shape[1]):
                        # print("K ",k)
                        # print("err ",err)
                        err = lamda * err + training_data[i][j][:,[k]]
                        # if current at terminal state A or G
                        if k == (training_data[i][j]).shape[1] - 1:
                            #equation 2
                            delta_weight += alpha*(reward[i][j] - np.dot(weight.transpose(), training_data[i][j][:,[k]])) * err
                        # if current at non terminal state
                        else:
                            #update delta
                            delta_weight += alpha*(np.dot(weight.transpose(),training_data[i][j][:,[k+1]]) - np.dot(weight.transpose(), training_data[i][j][:,[k]])) * err
                    delta_weight_acc += delta_weight
                # After no_seq or a set, update weight
                weight += delta_weight_acc
                #break loop when converge
                if (np.linalg.norm(previous_weight - weight) <= epsilon):
                    break
            
            rmse[lamda] += np.sqrt(np.mean((weight - ideal_weight)**2))
            # print("weight:",weight)
            # print("rmse: ",rmse[lamda])
    # print(rmse)
    plot_f3(rmse)

    #Figure 4: 
    #init list of alpha
    alpha_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    rmse2 = {}
    for lamda in lambda_list:
        rmse2[lamda] = {}
        for alpha in alpha_list:
            rmse2[lamda][alpha] = 0
    # print(rmse2)

    #Calculate rmse for experiment 2 for figure 4 and 5
    for lamda in lambda_list:
        for alpha in alpha_list:
            # print("lamda ",lamda)
            # print("alpha ",alpha)
            for i in range(no_set):
                #init for weight 
                weight = np.array([[0.5,0.5,0.5,0.5,0.5]]).transpose()
                # iterate through sequence
                for j in range(no_seq):
                    err = np.array([[0.0,0.0,0.0,0.0,0.0]]).transpose()
                    delta_weight = np.array([[0.0,0.0,0.0,0.0,0.0]]).transpose()
                    #iterate through each step
                    for k in range((training_data[i][j]).shape[1]):
                        # print("K ",k)
                        # print("err ",err)
                        err = lamda * err + training_data[i][j][:,[k]]
                        # if current at terminal state A or G
                        if k == (training_data[i][j]).shape[1] - 1:
                            #equation 2
                            delta_weight += alpha*(reward[i][j] - np.dot(weight.transpose(), training_data[i][j][:,[k]])) * err
                        # if current at non terminal state
                        else:
                            #update delta
                            delta_weight += alpha*(np.dot(weight.transpose(),training_data[i][j][:,[k+1]]) - np.dot(weight.transpose(), training_data[i][j][:,[k]])) * err
                    # After all step, update weight after each sequence
                    weight += delta_weight
                #Upd rmse after each set
                rmse2[lamda][alpha] += np.sqrt(np.mean((weight - ideal_weight)**2))
            # print("weight:",weight)
            # print("rmse2: ",rmse2[lamda][alpha])

    # print(rmse2)
    plot_f4(rmse2)

    plot_f5(rmse2)
