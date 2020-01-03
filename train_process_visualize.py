from tensorboardX import SummaryWriter
writer = SummaryWriter()
#注意，写完一定要close，Summary不添加参数的话，默认在cmd的命令是tensorboard --logdir runs
def acc_visualize(save_path,y_num,x_epoch,model = 1):
    writer.add_scalar(save_path,y_num,x_epoch)
    writer.close()
