"""Preprocesses Cornell Movie Dialog data."""
import nltk
import random
import tensorflow as tf

tf.app.flags.DEFINE_string("raw_data", "", "Raw data path")
tf.app.flags.DEFINE_string("train_file", "", "File to write train preprocessed data "
                                           "to.")
tf.app.flags.DEFINE_string("val_file", "", "File to write validate preprocessed data "
                                           "to.")
tf.app.flags.DEFINE_string("test_file", "", "File to write test preprocessed data "
                                           "to.")
FLAGS = tf.app.flags.FLAGS


def main(_):
    random.seed(111111)
    with open(FLAGS.raw_data, 'r', encoding='utf8', errors='ignore') as raw_data, \
            open(FLAGS.train_file, "w", encoding='utf8', errors='ignore') as train_file, \
            open(FLAGS.val_file, "w", encoding='utf8', errors='ignore') as val_file, \
            open(FLAGS.test_file, "w", encoding='utf8', errors='ignore') as test_file:
        for line in raw_data:
            parts = line.split(" +++$+++ ")
            dialog_line = parts[-1]
            s = dialog_line.strip().lower()
            preprocessed_line = " ".join(nltk.word_tokenize(s))
            score = random.random()
            if score<0.6:
                train_file.write((preprocessed_line + "\n"))
            elif score>=0.6 and score<0.8:
                val_file.write((preprocessed_line + "\n"))
            else:
                test_file.write((preprocessed_line + "\n"))

if __name__ == "__main__":
    tf.app.run()
