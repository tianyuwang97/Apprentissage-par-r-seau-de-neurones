import matplotlib.pyplot as plt
import numpy as np

def average_plot(fichier):
    x_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    mean_food_collected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mean_collision = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mean_death = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nb_sim = 0
    f = open(fichier+'.txt', "r")
    ligne = f.readline()
    while len(ligne) > 0:
        ligne = ligne[:-1]
        ligne = ligne.split(":")
        if ligne[0] == "nb_training":
            nb_sim += 1
            x_values += [int(s) for s in ligne[1][2:-1].split(',')]

        elif ligne[0] == "mean_collected_food":
            mean_food_collected += [float(s) for s in ligne[1][2:-1].split(',')]

        elif ligne[0] == "mean_collision":
            mean_collision += [float(s) for s in ligne[1][2:-1].split(',')]

        elif ligne[0] == "mean_death":
            mean_death +=[float(s) for s in ligne[1][2:-1].split(',')]

        ligne = f.readline()

    x_values = np.divide(x_values, nb_sim)
    mean_food_collected = np.divide(mean_food_collected, nb_sim)
    mean_collision = np.divide(mean_collision, nb_sim)
    mean_death = np.divide(mean_death, nb_sim)


    plt.figure("Mean food collected")
    plt.title("Mean food collected")
    plt.ylim(0, 13)
    plt.plot(x_values, mean_food_collected, "b:x", label="Average food eaten")
    plt.xlabel("Number of training iteration done")
    plt.ylabel("Food")
    plt.legend()
    plt.savefig(fichier+"_food")

    plt.figure("Mean deaths by enemies")
    plt.title("Mean deaths by enemies")
    plt.plot(x_values, mean_death, "b:x", label="Whithout collisions rewards")
    plt.xlabel("Number of training iteration done")
    plt.ylabel("Deaths")
    plt.legend()
    plt.savefig(fichier+"_death")

    plt.figure("Mean collisions")
    plt.title("Mean collisions")
    plt.plot(x_values, mean_collision, "g:x", label="Nb collisions")
    plt.xlabel("Number of training iteration done")
    plt.ylabel("Collisions")
    plt.legend()
    plt.savefig(fichier+"_collision")

    plt.show()

if __name__ == '__main__':
    average_plot("../Resultats/res")
