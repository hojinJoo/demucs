import matplotlib.pyplot as plt
import numpy as np

# generate some random data
def visualize(data,path):
    """Visualize the slots."""
    B,n_slots,n_src,Fq,T,n_channel = data.shape
    
    data = data.mean(axis=-1)
    print(data.size())
    # # create a figure with subplots
    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))

    # # plot each source in a separate subplot
    # for i in range(4):
    #     axs[i].plot(data[:, i])
    #     axs[i].set_title(f"Source {i+1}")

    # # set the title of the figure
    # fig.suptitle("16 Slots and 4 Sources")

    # # show the plot
    # plt.show()
    
    
