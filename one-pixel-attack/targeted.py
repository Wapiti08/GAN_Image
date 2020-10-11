import argparse
import logging

import imageio
import numpy as np
import yaml
from IPython import embed
from pathlib import Path

from common import get_image_array, get_probability_for_class, get_perturbed_images
from differential_evolution import init_population, gen_children
from models.base import get_model_from_name

CONFIG = None
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def fitness_function(prediction, target_class):
    """
    For targeted attacks, the fitness function is the probability of target class
    """
    return get_probability_for_class(prediction, target_class)


def get_fit_population(fathers, children, fathers_predictions, children_predictions, target_class):
    final_population = list()
    for i in range(len(fathers_predictions)):
        father_fitness = fitness_function(fathers_predictions[i], target_class)
        child_fitness = fitness_function(children_predictions[i], target_class)
        if father_fitness < child_fitness:
            final_population.append(children[i])
        else:
            final_population.append(fathers[i])
    return np.array(final_population)


def find_adversary_image(image, model, target_label):
    original_predictions = model.predict(np.copy(image))
    print(original_predictions)
    true_label = original_predictions[0][0][1]
    true_label_probability = original_predictions[0][0][2]

    logging.info("True label: {}, Probability: {}".format(true_label, true_label_probability))

    target_label_probability = get_probability_for_class(original_predictions[0], target_label)
    logging.info("Target label: {}, Probability: {}".format(target_label, target_label_probability))

    imageio.imwrite('output/original.jpg', image[0])

    population = init_population(CONFIG)
    for i in range(CONFIG["num_iterations"]):
        logging.info("Iteration: {}".format(i))
        perturbed_images = get_perturbed_images(image, population)
        perturbed_predictions = model.predict(np.copy(perturbed_images), top=model.num_classes)

        target_class_probabilities = map(lambda p: get_probability_for_class(p, target_label), perturbed_predictions)
        logging.info("Probabilites for target class: Min={}, Max={}".format(min(target_class_probabilities),
                                                                            max(target_class_probabilities)))
        if i % 10 == 0:
            imageio.imwrite('output/{}.jpg'.format(i),
                            perturbed_images[target_class_probabilities.index(max(target_class_probabilities))])

        population_children = gen_children(population, CONFIG)
        perturbed_images_children = get_perturbed_images(image, population_children)
        perturbed_predictions_children = model.predict(np.copy(perturbed_images_children), top=model.num_classes)

        population = get_fit_population(population, population_children,
                                        perturbed_predictions,
                                        perturbed_predictions_children,
                                        target_class=target_label)
    embed()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-c', dest='config_file', help='config file')
    # parser.add_argument('--input', '-i', dest='input_image', help='input image file')
    # parser.add_argument('--target', '-t', dest='target_class', help='target class name')
    # args = parser.parse_args()


    # ===================== Test for new dataset ====================
    # create the current path
    current_path = Path(__file__).parent.parent.absolute()
    config_file = current_path/'one-pixel-attack'/'config.yaml'

    CONFIG = yaml.safe_load(open(config_file.as_posix()))
    model = get_model_from_name(CONFIG["model"])
    
    # read image from outside image folder
    input_folder = current_path/'Images'/'test'
    # read pair.txt
    pair_file = current_path/input_folder/'pair.txt'
    # generate the map dictionary
    map_png = {}
    with pair_file.open() as fr:
        pair = fr.readlines()
        for line in pair:
            map_png[line.split(" ")[0]]=line.split(" ")[1].strip()
    
    
    target_labels = map_png.values()
    CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size
    
    for attack_image, target_image in map_png.items():
        # create the attack image path
        attack_image_path = input_folder/attack_image
        # create the target image path
        target_image_path = input_folder/target_image
        # generate the target fake images
        image_arr = get_image_array(attack_image_path.as_posix(), config=CONFIG)

        find_adversary_image(image_arr, model, target_label)

'''
from pathlib import Path
current_path = Path(__file__).parent.absolute()
print(current_path)

for path in current_path.glob('*.py'):
    print(path)
'''