import os, logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils as U


class Visualizer():
    def __init__(self, args):
        self.args = args
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['adc'] = [
            'Normal Forward Driving 0', 'Drinking 1', 'Phone Call(right) 2', 'Phone Call(left) 3', 'Eating 4', 'Text (Right) 5',
            'Text (Left) 6', 'Hair / makeup 7', 'Reaching behind 8', 'Adjust control panel 9', 'Pick up from floor (Driver) 10', 'Pick up from floor (Passenger) 11',
            'Talk to passenger at the right 12', 'Talk to passenger at backseat 13', 'yawning 14', 'Hand on head 15', 'Singing with music 16',
            'shaking or dancing with music 17'
        ]

        self.font_sizes = {
            'adc': 6,
        }


    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) heatmap (hm)')
        logging.info('4) confusion matrix (cm), 5) action accuracy (ac)')
        logging.info('Please input the number (or name) of the function, q for quit: ')
        while True:
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 'hm' or cmd == 'heatmap':
                self.show_heatmap()
            elif cmd == '4' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '5' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')


    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        if os.path.exists(data_file):
            data = np.load(data_file)
        else:
            data = None
            logging.info('')
            logging.error('Error: Do NOT exist this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        logging.info('*********************Video Name************************')
        logging.info(data['name'][self.args.visualization_sample])
        logging.info('')

        feature = data['feature'][self.args.visualization_sample,:,:,:,:]
        self.data = data['data'][self.args.visualization_sample,:,:,:,:]
        self.label = data['label']
        weight = data['weight']
        out = data['out']
        cm = data['cm']
        self.cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

        dataset = self.args.dataset.split('-')[0]
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class-1]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)   # CAM method
        self.result = self.result[self.args.visualization_class-1,:,:,:]


    def show_action_accuracy(self):
        cm = self.cm.round(4)

        logging.info('Accuracy of each class:')
        accuracy = cm.diagonal()
        for i in range(len(accuracy)):
            logging.info('{}: {}'.format(self.names[i], accuracy[i]))
        logging.info('')

        plt.figure()
        plt.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.show()


    def show_confusion_matrix(self):
        cm = self.cm.round(2)
        show_name_x = range(1,len(self.names)+1)
        show_name_y = self.names

        plt.figure()
        font_size = self.font_size
        sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, annot_kws={'fontsize':font_size-2}, cbar=False,
                    square=True, linewidths=0.1, linecolor='black', xticklabels=show_name_x, yticklabels=show_name_y)
        plt.xticks(fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Index of Predict Classes', fontsize=font_size)
        plt.ylabel('Index of True Classes', fontsize=font_size)
        plt.savefig('cfm')
        plt.show()

    def show_heatmap(self):
        I, C, T, V, M = self.data.shape
        max_frame = T
        for t in range(T):
            if np.sum(self.data[:,:,t,:,:]) == 0:
                max_frame = t
                break

        plt.figure(1)
        plt.gcf().subplots_adjust(right=0.8)  # Create room on the right

        skeleton1 = self.result[:,:,0]
        heat1 = np.zeros((max_frame//4*4, V))
        for t in range(max_frame//4):
            d1 = (skeleton1[t+1,:] - skeleton1[t,:]) / 4
            for i in range(4):
                heat1[t*4+i,:] = skeleton1[t,:] + d1 * i

        plt.subplot(211)
        plt.imshow(heat1.T, cmap=plt.cm.plasma, vmin=0, vmax=np.max(heat1))
        plt.ylabel('Joints')
        plt.xlabel('Frames')
        plt.title('Person 1')

        if self.result.shape[-1] > 1:
            skeleton2 = self.result[:,:,1]
            heat2 = np.zeros((max_frame//4*4, V))
            for t in range(max_frame//4):
                d2 = (skeleton2[t+1,:] - skeleton2[t,:]) / 4
                for i in range(4):
                    heat2[t*4+i,:] = skeleton2[t,:] + d2 * i

            plt.subplot(212)
            plt.imshow(heat2.T, cmap=plt.cm.plasma, vmin=0, vmax=np.max(heat1))
            plt.ylabel('Joints')
            plt.title('Person 2')

        plt.xlabel('Frames')
        plt.colorbar(cax=plt.gcf().add_axes([0.85, 0.1, 0.05, 0.8]))
        plt.show()


    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample**********************')
        logging.info(wrong_sample)
        logging.info('')


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum)
        logging.info('')
        logging.info('Most important joints:')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:,:,1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum)
            logging.info('')
            logging.info('Most important joints:')
            logging.info(second_index)
            logging.info('')