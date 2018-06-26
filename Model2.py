import tensorflow as tf
import numpy as np
import gym
import argparse
import time

print("importing done")


def fc_layer(inp, in_size, out_size, activation_function=None, name="fc"):
    with tf.name_scope(name):
        W = tf.get_variable(name='W', shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(shape=[out_size], name='b', initializer=tf.contrib.layers.xavier_initializer())
        activation = tf.matmul(inp, W) + b

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)

        if activation_function == 'tanh':
            out = tf.nn.tanh(activation, name='tanh_activation')
        elif activation_function == 'sigmoid':
            out = tf.nn.sigmoid(activation, name='sigmoid_activation')
        elif activation_function == 'relu':
            out = tf.nn.relu(activation, name='relu_activation')
        else:
            return activation

        tf.summary.histogram("outputs", out)
        return out


def discounted_rewards(rew, gamma=0.99):
    rewards = np.zeros_like(rew)
    temp = 0
    for i in reversed(range(len(rew))):
        if rew[i] != 0:
            temp = 0
        temp += rew[i]
        rewards[i] = temp
        temp *= gamma
    return rewards


def model(X, observation_size, num_actions):
    flat = tf.layers.flatten(X)  # to deal with the case if the input is an image
    # TODO use convolutional layers where observations are images or define a diff model in that case.
    fc1 = fc_layer(flat, observation_size, 128, activation_function='relu', name='fc1')
    fc2 = fc_layer(fc1, 128, 256, activation_function='relu', name='fc2')
    fc3 = fc_layer(fc2, 256, 128, activation_function='relu', name='fc3')
    fc4 = fc_layer(fc3, 128, num_actions, activation_function='sigmoid', name='fc4')

    return fc4


def run(args):
    if not (args.test or args.train):
        print('nothing to be done')
        return

    # extracting info about the environment
    env = gym.make(args.model_id)
    num_actions = env.action_space.n
    observation_size = np.prod(env.observation_space.shape)
    print("action space size = " + str(num_actions))
    print("observation space size = " + str(observation_size))

    tf.reset_default_graph()

    # creating placeholders
    observation_placeholder = tf.placeholder(tf.float32, shape=[None, observation_size], name='observations')
    action_placeholder = tf.placeholder(tf.int32, shape=[None], name='actions')
    reward_placeholder = tf.placeholder(tf.float32, shape=[None], name='rewards')
    # tf.summary.scalar('reward', reward_placeholder)  # TODO see if correct

    model_output = model(observation_placeholder, observation_size, num_actions)

    # a function to sample actions from previously learnt model
    with tf.name_scope('sampling'):
        sample_action = tf.squeeze(tf.multinomial(tf.reshape(model_output, [1, num_actions]), num_samples=1))

    with tf.name_scope('train'):
        one_hot = tf.one_hot(action_placeholder, num_actions)
        cross_entropy = - tf.reduce_mean(tf.multiply(one_hot, tf.log(model_output)) +
                                        tf.multiply(1-one_hot, tf.log(1 - model_output)), axis=1)
        # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.one_hot(action_placeholder, num_actions), logits=model_output), axis=1)
        loss = tf.reduce_mean(tf.multiply(reward_placeholder, cross_entropy))
        tf.summary.scalar('loss', loss)
        train_operation = tf.train.AdamOptimizer().minimize(loss)

    # TODO add rewards scalar to the summary
    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.save_dir)
            saver.restore(sess, restore_path)
            print('restored the model from' + str(restore_path))
        else:
            sess.run(init)

        if args.train:
            summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

            print("Training Started")
            training_start = time.time()

            for epoch in range(args.num_epochs):
                epoch_start = time.time()
                training_data = []
                rewards = []

                # data collection
                while len(training_data) < args.batch_size:
                    # curr_rewards = []
                    observation = env.reset()
                    for t in range(args.max_timesteps):
                        action = sess.run(sample_action, feed_dict={observation_placeholder: [observation]})
                        training_data.append([observation, action])
                        observation, reward, done, info = env.step(action)
                        # curr_rewards.append(reward)
                        # rewards.append(reward)
                        if done:
                            rewards.append(1)
                            break
                        else:
                            rewards.append(0)
                    # discounted_rewards(curr_rewards, gamma=args.gamma)

                # discount rewards
                rewards = discounted_rewards(rewards, gamma=args.gamma)
                # normalize rewards
                rewards = (rewards - np.mean(rewards))/np.std(rewards)

                print(rewards)

                feed_dict = {observation_placeholder: np.array([i[0] for i in training_data]).reshape(-1, observation_size),
                             action_placeholder: np.array([int(i[1]) for i in training_data]),
                             reward_placeholder: rewards}
                _loss, _, _summary = sess.run([loss, train_operation, summary], feed_dict=feed_dict)

                summary_writer.add_summary(_summary)

                print("Epoch " + str(epoch + 1) +
                      " completed in " + str(time.time() - epoch_start)[:5] +
                      " secs with loss " + str(_loss))
                # print("max reward = " + str(np.max(rewards)) +
                #       " average rewards = " + str(np.average(rewards)))

            save_path = saver.save(sess, args.save_dir + 'model.ckpt')
            print("Training Finished in " + str(time.time() - training_start)[:5])
            print("model saved in the dir " + str(save_path))

        if args.test:
            rewards = []

            for episode in range(args.test_episodes):
                obs = env.reset()
                actions = []
                total_reward = 0
                for t in range(args.max_timesteps):
                    env.render()
                    action = np.argmax(sess.run(model_output, feed_dict={observation_placeholder: [obs]}))
                    actions.append(action)
                    observation, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        print(str(episode+1) + " episode finished after " + str(t + 1) + " timesteps")
                        rewards.append(total_reward)
                        print(actions)
                        break

                print("average action = " + str(np.average(actions)) +
                      " total reward = " + str(total_reward))

            print("Finished")
            print("max reward = " + str(np.max(rewards)) +
                  " average rewards = " + str(np.average(rewards)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('openAI')
    parser.add_argument('--model_id', type=str, default='CartPole-v0')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--max_timesteps', type=int, default=200)
    parser.add_argument('--test_episodes', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--save_dir', type=str,
                        default="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\saves\\")
    parser.add_argument('--summary_dir', type=str,
                        default="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\temp\\")

    args = parser.parse_args()

    run(args)

# tensorboard --logdir="C:\\My Folder\\Programming\\Deep Learning\\New folder\\Open AI Cartpole\\temp\\"

