import tensorflow as tf
import numpy as np
import gym
import argparse
import time

print("importing done")


def fc_layer(input, in_size, out_size, name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='b')
        activation = tf.matmul(input, W) + b
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return activation


def discounted_rewards(r, gamma):
    rewards = np.zeros_like(r)
    temp = 0
    for i in reversed(range(len(r))):
        if r[i] != 0:
            temp = 0
        temp += r[i]
        rewards[i] = temp
        temp *= gamma
    return rewards


def model(args):
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, 4], name='X')  # observation
    Y = tf.placeholder(tf.int32, shape=[None], name='Y')  # action
    R = tf.placeholder(tf.float32, shape=[None], name='R')  # reward

    # using two layers
    fc1 = tf.nn.relu(fc_layer(X, 4, 128, name='fc1'))
    fc2 = tf.nn.relu(fc_layer(fc1, 128, 256, name='fc2'))
    fc3 = tf.nn.relu(fc_layer(fc2, 256, 128, name = 'fc3'))
    output_layer = fc_layer(fc3, 128, 2, name='output_layer')  # output of the network

    with tf.name_scope('sampling'):
        # tf.multinomial -> Draws samples from a multinomial distribution.
        # tf.squeeze -> Removes dimensions of size 1 from the shape of a tensor.
        sample_action = tf.squeeze(tf.multinomial(logits=tf.reshape(output_layer, [1, 2]), num_samples=1))

    with tf.name_scope('train'):
        # print(fc2)
        # print(Y)
        cross_entropy = tf.losses.softmax_cross_entropy(tf.one_hot(Y, 2), output_layer)
        loss = tf.reduce_sum(R*cross_entropy)
        tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer().minimize(loss)

    summary = tf.summary.merge_all()

    env = gym.make('CartPole-v0')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.save_dir)
            print("restoring from " + str(restore_path))
            saver.restore(sess, restore_path)
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        print("Training")
        for epoch in range(args.num_epochs):
            start_time = time.time()
            observations = []
            actions = []
            rewards = []
            while len(observations) < args.batch_size:
                observation = env.reset()
                for t in range(args.max_timesteps):
                    if args.render:
                        env.render()
                    action = sess.run(sample_action, feed_dict={X: [observation]})
                    actions.append(action)
                    observations.append(observation)
                    observation, reward, done, info = env.step(action)
                    rewards.append(reward)
                    if done:
                        print("episode finished after " + str(t + 1) + " timesteps")
                        break

            print(rewards)
            rewards = discounted_rewards(rewards, args.gamma)
            print(rewards)
            # rewards -= np.mean(rewards)
            # rewards /= np.std(rewards)
            _loss, _, _summary = sess.run([loss, train_op, summary],
                                          feed_dict={X: observations, Y: actions, R: rewards})
            summary_writer.add_summary(_summary)
            save_path = saver.save(sess, args.save_dir + 'model.ckpt')

            print("Epoch " + str(epoch + 1) +
                  " completed in " + str(time.time() - start_time)[0:4] +
                  " secs with loss " + str(_loss))
            print("max reward = " + str(np.max(rewards)) +
                  " average rewards = " + str(np.average(rewards)))
            # print('Model checkpoint saved ' + str(save_path))

        print("Training Finished")

        if args.test:
            rewards = []
            for episode in range(10):
                actions = []
                observation = env.reset()
                tot_rew = 0
                for t in range(args.max_timesteps):
                    env.render()
                    action = np.argmax(sess.run([output_layer], feed_dict={X: [observation]}))
                    actions.append(action)
                    print(action)
                    observation, reward, done, info = env.step(action)
                    tot_rew += reward
                    if done:
                        print("{} episode finished after {} timesteps".format(episode + 1, t + 1))
                        break
                rewards.append(tot_rew)
                print("average action = " + str(np.average(actions)) +
                      " total reward = " + str(tot_rew))
            print("FINISHED")
            print("max reward = " + str(np.max(rewards)) +
                  " average rewards = " + str(np.average(rewards)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cartpole-v0')
    parser.add_argument('--test', type=str, default=False)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--max_timesteps', type=int, default=200)
    parser.add_argument('--save_dir', type=str,
                        default="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\saves\\")
    parser.add_argument('--summary_dir', type=str,
                        default="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\temp\\")

    args = parser.parse_args()

    model(args)

## tensorboard --logdir="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\temp\\"
