import numpy as np
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt




if __name__ == "__main__":

    wav_root_path = "/data1/gs/SVS_system/egs/public_dataset/db_joint/exp/3_18_augment/train_result"

    singer_correct_map = np.zeros((7,7)).astype(np.float64)
    singer_total_map = np.zeros((7,7)).astype(np.float64)
    phone_correct_map = np.zeros((7,7)).astype(np.float64)
    phone_total_map = np.zeros((7,7)).astype(np.float64)
    semitone_correct_map = np.zeros((7,7)).astype(np.float64)
    semitone_total_map = np.zeros((7,7)).astype(np.float64)

    fwrite = open("filter_wav_filename.txt","w")
    num = 0
    with open("filter_res.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            filename, singer_predict, phone_correct, semitone_correct, valid_length = line.strip().split("|")

            db_name = filename.split("_")[0]
            if db_name == "hts":
                singer_from = 0
            elif db_name == "jsut":
                singer_from = 1
            elif db_name == "kiritan":
                singer_from = 2
            elif db_name == "natsume":
                singer_from = 3
            elif db_name == "pjs":
                singer_from = 4
            elif db_name == "ofuton":
                singer_from = 5
            elif db_name == "oniku":
                singer_from = 6
            singer_to = int( filename.split("-")[1] )

            singer_total_map[singer_from, singer_to] += 1
            phone_total_map[singer_from, singer_to] += int(valid_length)
            semitone_total_map[singer_from, singer_to] += int(valid_length)

            if int(singer_predict) == singer_to:
                singer_correct_map[singer_from, singer_to] += 1
            phone_correct_map[singer_from, singer_to] += int(phone_correct)
            semitone_correct_map[singer_from, singer_to] += int(semitone_correct)

            if int(singer_predict) == singer_to and int(phone_correct) / int(valid_length) >= 0.8 and int(semitone_correct) / int(valid_length) >= 0.9  :
                num += 1
                write_path = os.path.join(wav_root_path, filename+".wav")
                fwrite.write(f"{write_path}\n")
    fwrite.close()          

    singer_acc_map = singer_correct_map / singer_total_map
    phone_acc_map = phone_correct_map / phone_total_map
    semitone_acc_map = semitone_correct_map / semitone_total_map

    print(f"singer_acc_map: {singer_acc_map}")
    print(f"phone_acc_map: {phone_acc_map}")
    print(f"semitone_acc_map: {semitone_acc_map}")


    sns_plot1 = sns.heatmap(singer_acc_map, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    sns_plot1.set(xlabel='singer_to', ylabel='singer_from', title='singer_acc_map')
    # sns_plot1.figure.savefig("singer_acc_map.png")
    plt.savefig('singer_acc_map.png')
    plt.clf()

    sns_plot2 = sns.heatmap(phone_acc_map, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    sns_plot2.set(xlabel='singer_to', ylabel='singer_from', title='phone_acc_map')
    # sns_plot2.figure.savefig("phone_acc_map.png")
    plt.savefig('phone_acc_map.png')
    plt.clf()

    sns_plot3 = sns.heatmap(semitone_acc_map, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    sns_plot3.set(xlabel='singer_to', ylabel='singer_from', title='semitone_acc_map')
    # sns_plot3.figure.savefig("semitone_acc_map.png")
    plt.savefig('semitone_acc_map.png')

    print(f"num: {num}")
    