import numpy as np
import collections
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch


# Hyperparameters
learning_rate = 0.005
gamma = 0.9  # 감쇠인자 강화학습이 미래의 가치를 고려해서 현재 보상에 ~ 얼마나 반영할 지
n_episode =300000
buffer_limit = 10000
batch_size = 16

class grid_world():
    def __init__(self):
        self.robot_1_pos = np.array([1, 0])
        self.robot_2_pos = np.array([3, 0])
        self.r1_goal = np.array([0, 0])
        self.r2_goal = np.array([0, 0])
        self.goal_num = 0
        self.robot_1_on = True
        self.robot_2_on = True
        self.rotary_status = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.rotary_pos = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        self.grid = [
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]
        ]
        self.goal_pos = np.array([[0, 1], [0, 3], [1, 4], [3, 4], [4, 1], [4, 3]])
        self.action_list = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])
    def set_r1_goal(self, a):
        self.r1_goal = self.goal_pos[a]

    def set_r2_goal(self, a):
        self.r2_goal = self.goal_pos[a]

    def r1_is_done(self, curr_pos):
        r = 0
        if np.array_equal(self.r1_goal , curr_pos):
            done = True
            r = 30
            self.robot_1_on = False
        else:
            done = False
        return done, r

    def r2_is_done(self, curr_pos):
        r = 0
        if np.array_equal(self.r2_goal , curr_pos):
            done = True
            r = 30
            self.robot_2_on = False
        else:
            done = False
        return done, r

    def step_r1(self, a):
        done_obj = False
        self.robot_1_pos += self.action_list[a]
        if 0 <= self.robot_1_pos[0] <= 4 and 0 <= self.robot_1_pos[1] <= 4:
            # 그리드 조건 충족?
            if self.grid[self.robot_1_pos[0]][self.robot_1_pos[1]] == 0:
                pass
                r = -1  # 최단 경로
            else:
                done_obj = True
                r = -20
        else:
            done_obj = True
            r = -20
        # 충돌 조건
        if np.array_equal(self.robot_1_pos, self.robot_2_pos):
            done_obj = True
            r = -20
        else:
            pass
        # 종료 조건
        done, r_arrived = self.r1_is_done(self.robot_1_pos)
        r += r_arrived
        s = np.array([self.robot_1_pos[0]- self.r1_goal[0], self.robot_1_pos[1] - self.r1_goal[1],
                      self.robot_1_pos[0]- self.robot_2_pos[0], self.robot_1_pos[1]-self.robot_2_pos[1],
                      self.robot_2_pos[0] - self.r2_goal[0], self.robot_2_pos[1] - self.r2_goal[1]])
        return s, r, done, done_obj


    def step_r2(self, a):
        done_obj = False
        r_arrived = 0
        self.robot_2_pos += self.action_list[a]
        # 그리드 밖으로 안나감?
        if 0 <= self.robot_2_pos[0] <= 4 and 0 <= self.robot_2_pos[1] <= 4:
            # 그리드 조건 충족?
            if self.grid[self.robot_2_pos[0]][self.robot_2_pos[1]] == 0:
                pass
                r = -1  # 최단 경로
            else:
                done_obj = True
                r = -20
        else:
            done_obj = True
            r = -20
        # 충돌 조건
        if np.array_equal(self.robot_1_pos, self.robot_2_pos):
            done_obj = True
            r = -20
        else:
            pass
        # 종료 조건
        done, r_arrived = self.r2_is_done(self.robot_2_pos)
        r += r_arrived
        s = np.array([self.robot_2_pos[0]- self.r2_goal[0], self.robot_2_pos[1] - self.r2_goal[1],
                      self.robot_2_pos[0]- self.robot_1_pos[0], self.robot_2_pos[1]-self.robot_1_pos[1],
                      self.robot_1_pos[0] - self.r1_goal[0], self.robot_1_pos[1] - self.r1_goal[1]])
        return s, r, done, done_obj

    def reset(self):
        self.robot_1_on = True
        self.robot_2_on = True
        self.robot_1_pos = np.array([1, 0])
        self.robot_2_pos = np.array([3, 0])
        # 0~5 명령 위치 설정
        self.goal_num = random.sample(range(0, 6), 2)
        self.set_r1_goal(self.goal_num[0])
        self.set_r2_goal(self.goal_num[1])
        self.rotary_status = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        s = np.array([self.robot_1_pos[0]- self.r1_goal[0], self.robot_1_pos[1] - self.r1_goal[1],
                      self.robot_2_pos[0]- self.robot_1_pos[0], self.robot_2_pos[1]-self.robot_1_pos[1],
                      self.robot_2_pos[0] - self.r2_goal[0], self.robot_2_pos[1] - self.r2_goal[1]])
        return s

    def is_odd(self, num):
        if num % 2 != 0:
            return True
        else:
            return False

    def send_change(self, pos, command):
        """
        로터리 변환 신호 주는 함수
        pos : 로터리 위치
        command : 바꿀지 말지 (True : 바꾸면 됨, False : 바꾸지 말고
        """
        if command == True:
            # 원하는 동작 넣는 곳 (로터리 변환)
            pass
        else:
            pass

        # return signal

    def change_rotary(self, robot_pos, a):
        """
        rotary의 상태는 0 : 세로
                      1 : 가로로 정의한다.
                      2 : 변환 중
        a는 action 행동 ex) [1, 0] y축 좌표로 1칸 이동

        robot_pos 입력받은 로봇의 위치

        경우의 수
            rotary 0 action [0, 1]
            action[1] - rotary = 1 (불가능)
            rotary 1 action [0, 1]
            action[1] - rotary = 0 (가능)
            rotary 0 action [1, 0]
            action[1] - rotary = 0 (가능)
            rotary 1 action [1, 0]
            action[1] - rotary = -1 (불가능)
            rotary 0 action [-1, 0]
            action[1] - rotary = 0 (가능)
            rotary 1 action [-1, 0]
            action[1] - rotary = -1 (불가능)
            rotary 0 action [0, -1]
            action[1] - rotary = -1 (불가능)
            rotary 1 action [0, -1]
            action[1] - rotary = -2 (가능)

            뺀 값 만큼 로터리에 더해주면 됨

        경우의 수 두 번째
            가로로 움직여서 로터리 위치
            세로로 움직여서 로터리 위치
            a[1, 0]일 때 로터리 1 False
            a[0, 1]일 때 로터리 0

        """
        # 완수 신호
        signal = False

        # 현재 로터리 위인가?
        if self.rotary_pos[robot_pos[0]][robot_pos[1]] == 1:
            # 조정할 로터리
            change_rot_pos = self.rotary_pos[robot_pos[0]][robot_pos[1]]
            change_rot = a[1] - self.rotary_status[robot_pos[0]][robot_pos[1]]
        else:
            rotary = robot_pos + a
            # 이동 후 로터리 위에 위치하게 되는가?
            if self.rotary_pos[rotary[0]][rotary[1]] == 1:
                change_rot_pos = self.rotary_pos[robot_pos[0]][robot_pos[1]]
                change_rot = a[1] - self.rotary_status[robot_pos[0]][robot_pos[1]]
            else:
                pass

        if self.is_odd(change_rot) == True:
            change_flag = True
        else:
            change_flag = False

        # 만약 위의 함수가 완료되면 변경
        # signal = self.send_change(change_rot_pos, change_flag)
        signal = True

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # double-ended queue 양방향에서 데이터를 처리할 수 있는 queue형 자료구조를 의미한다

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    """
    action 네트워크에서 출력 2개 정수부만 사용 0이면 -> a = [1,0]  1이면 -> [0, 1]
    """
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 3)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)       # q_out tensor에서 a 자리에 있는 열들 중 a값에 해당하는 위치를 인덱싱해서 뽑아옴
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask   #
        loss = F.smooth_l1_loss(q_a, target)   # smooth_l1_loss 함수는 Huber loss 함수와 같음
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    save_directory = f"{os.getcwd()}{os.sep}params"
    os.makedirs(save_directory, exist_ok=True)  # 경로 없을 시  생성
    env = grid_world()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()


    print_interval = 20
    score = 0.0
    count = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)     # optimizer 설정 Adam 사용

    for n_epi in range(n_episode):
        epsilon = max(0.01, 0.2 - 0.001 * n_epi / 1000)
        s = env.reset()
        done = False
        while not done:
            if env.robot_1_on == True:
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done_1, done_obj = env.step_r1(a)
                done_mask = 0.0 if done_1 else 1.0
                memory.put((s, a, r, s_prime, done_mask))
                score += r
                s = s_prime
            else:
                done_1 = True

            if done_obj == True:
                break

            if env.robot_2_on == True:
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done_2, done_obj = env.step_r2(a)
                done_mask = 0.0 if done_2 else 1.0
                memory.put((s, a, r, s_prime, done_mask))
                score += r
                s =s_prime

            if done_obj == True:
                break

            if done_1 and done_2:
                count += 1
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())    # q_target 업데이트 20번에 한번 씩
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f} success : {}%".format(
                n_epi, score/print_interval, memory.size(), epsilon * 100, count / print_interval * 100))

            params = q.state_dict()
            file_name = str(n_epi % print_interval) + "param.pt"
            file_path = os.path.join(save_directory, file_name)
            torch.save(params, file_path)
            score = 0.0
            count = 0

if __name__ == '__main__':
    main()
