import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

#-Classes for the environment-----------------------------------------------------

class Boundary:
    def __init__(self, x1, y1, x2, y2):
        self.a = np.array([float(x1), float(y1)])
        self.b = np.array([float(x2), float(y2)])

    def display(self, screen):
        pygame.draw.line(screen, "white", (self.a[0], self.a[1]), 
                         (self.b[0], self.b[1]), 10)

class RewardGate:
    gates_left = 0
    gates_total = 0
    gates_passed = 0

    def __init__(self, x1, y1, x2, y2):
        self.a = np.array([float(x1), float(y1)])
        self.b = np.array([float(x2), float(y2)])
        self.active = True
        type(self).gates_total += 1
        type(self).gates_left += 1

    def set_active(self):
        self.active = True

    def display(self, screen):
        if self.active:
            pygame.draw.line(screen, "green", (self.a[0], self.a[1]), 
                             (self.b[0], self.b[1]), 10)
    
    def pass_gate(self):
        if self.active:
            self.active = False
            type(self).gates_left -= 1
            type(self).gates_passed += 1


class Ray:
    def __init__(self, pos, angle):
        self.pos = pos
        self.dir = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    def display(self, screen):
        pygame.draw.line(screen, "white", (self.pos[0], self.pos[1]),
                            (self.pos[0] + self.dir[0] * 100, 
                             self.pos[1] + self.dir[1] * 100), 1)

    def update(self, pos, angle):
        self.pos = pos
        self.dir = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    def cast(self, wall):
        x1 = wall.a[0]
        y1 = wall.a[1]
        x2 = wall.b[0]
        y2 = wall.b[1]

        x3 = self.pos[0]
        y3 = self.pos[1]
        x4 = self.pos[0] + self.dir[0]
        y4 = self.pos[1] + self.dir[1]

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))/den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))/den
        if t > 0 and t < 1 and u > 0:
            pt = np.array([0.0, 0.0])
            pt[0] = x1 + t * (x2 - x1)
            pt[1] = y1 + t * (y2 - y1)
            return pt
        else:
            return
        
    def get_distance(self, boundary):
        closest = None
        record = np.inf
        if isinstance(boundary, list):
            for wall in boundary:
                pt = self.cast(wall)
                if pt is not None:
                    d = np.linalg.norm(self.pos - pt)
                    if d < record:
                        record = d
                        closest = pt
            if closest is not None:
                return record
            else:
                return 2000.0
        else:
            pt = self.cast(boundary)
            if pt is not None:
                d = np.linalg.norm(self.pos - pt)
                if d < record:
                    record = d
                    closest = pt
        if closest is not None:
            return record
        else:
            return 2000.0
        
class Car:
    def __init__(self, x, y, rotation = 0.0, image = None):
        self.start_pos = np.array([float(x), float(y)])
        self.pos = np.array([float(x), float(y)])
        self.start_rotation = rotation
        self.rotation = rotation
        if image is not None:
            self.image = pygame.image.load(image)
            self.image = pygame.transform.scale(self.image, (30, 60))
            self.sprite = self.image.get_rect()
        else:
            self.image = None
            self.sprite = None
        self.turn_speed = 5.0
        self.max_speed = 10.0
        
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.max_acceleration = 0.8
        self.friction = 0.2

        self.rays = []
        self.num_rays = 12
        for a in range(0, 360, 360//self.num_rays):
            self.rays.append(Ray(self.pos, a))
        
        self.destroyed = False

    def set_image(self, image):
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image, (30, 60))
        self.sprite = self.image.get_rect()

    def display(self, screen):
        if self.image is not None:
            updated_image = pygame.transform.rotate(self.image, -self.rotation-90)
            self.sprite = updated_image.get_rect(center = self.pos)
            screen.blit(updated_image, self.sprite)

    def display_rays(self, track, screen):
        for ray in self.rays:
            closest = None
            record = np.inf
            for wall in track:
                pt = ray.cast(wall)
                if pt is not None:
                    d = np.linalg.norm(self.pos - pt)
                    if d < record:
                        record = d
                        closest = pt
            if closest is not None:
                pygame.draw.line(screen, "white", (ray.pos[0], ray.pos[1]),
                                    (closest[0], closest[1]), 1)

    def get_distances(self, track):
        distances = []
        for ray in self.rays:
            distances.append(ray.get_distance(track))
        return np.array(distances)

    def check_collision(self, track):
        for i in range(0, self.num_rays, self.num_rays//4):
            if self.rays[i].get_distance(track) < 20.0:
                return True

    def check_gate_pass(self, reward_gates):
        for gate in reward_gates:
            if gate.active:
                for i in range(0, self.num_rays, self.num_rays//4):
                    if self.rays[i].get_distance(gate) < 20.0:
                        gate.pass_gate()
                        return True

    def reset(self):
        self.pos = self.start_pos.copy()
        self.rotation = self.start_rotation
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])

    def move_car(self, direction):
        if direction == 'forward':
            force = np.array([np.cos(np.radians(self.rotation)), 
                              np.sin(np.radians(self.rotation))])
            self.acceleration += force * self.max_acceleration
        elif direction == 'backward':
            force = np.array([np.cos(np.radians(self.rotation)), 
                              np.sin(np.radians(self.rotation))])
            self.acceleration -= force * self.max_acceleration
        elif direction == 'left':
            self.rotation -= self.turn_speed
        elif direction == 'right':
            self.rotation += self.turn_speed

    def update(self, track):
        self.velocity += self.acceleration
        if (np.linalg.norm(self.acceleration) == 0):
            self.velocity *= 1 - self.friction
        
        if (np.linalg.norm(self.velocity) > self.max_speed):
            self.velocity = self.max_speed * self.velocity / np.linalg.norm(self.velocity)

        self.pos += self.velocity

        self.acceleration = np.array([0.0, 0.0])

        for a in range(0, 360, 360//self.num_rays):
            self.rays[a//(360//self.num_rays)].update(self.pos, self.rotation + a)

        if self.check_collision(track):
            self.destroyed = True

#-Create the environment--------------------------------------------------------

class CarEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.width = 1800
        self.height = 1000
        self.time_passed = 0
        self.max_time = 1000

        self.car = Car(710, 885, 0)
        self.background = None

        self.track_outer_cor = [
            (794, 953), (921, 945), (1021, 917), (1119, 851), (1170, 759), (1182, 649), 
            (1200, 605), (1267, 578), (1387, 563), (1504, 509), (1561, 405), (1574, 281), 
            (1535, 165), (1455, 87), (1319, 47), (1182, 35), (986, 50), (879, 105), (807, 237), 
            (785, 308), (714, 357), (620, 375), (462, 373), (336, 387), (239, 443), (183, 517), 
            (153, 639), (171, 761), (219, 851), (318, 917), (452, 947), (794, 953)
        ]

        self.track_inner_cor = [
            (794, 825), (890, 815), (975, 795), (1043, 735), (1053, 625), (1083, 533), 
            (1134, 490), (1221, 455), (1347, 437), (1413, 411), (1443, 327), (1427, 239), 
            (1344, 185), (1212, 167), (1034, 176), (981, 195), (939, 277), (861, 427), (785, 479), 
            (680, 509), (425, 507), (315, 573), (290, 656), (318, 749), (425, 818), (794, 825)
        ]

        self.reward_gates_cor = [
            (792, 968), (792, 781), (894, 963), (879, 779), (993, 943), (953, 767), (1062, 917), 
            (994, 747), (1134, 859), (1011, 733), (1176, 779), (1031, 717), (1187, 707), (1030, 683), 
            (1191, 665), (1039, 603), (1204, 631), (1068, 521), (1228, 617), (1137, 463), (1256, 593), 
            (1209, 433), (1299, 583), (1284, 424), (1373, 578), (1337, 419), (1473, 548), (1378, 395), 
            (1539, 481), (1404, 381), (1573, 395), (1407, 363), (1584, 326), (1421, 332), (1575, 243), 
            (1418, 287), (1548, 175), (1406, 257), (1507, 105), (1387, 239), (1426, 59), (1359, 211), 
            (1350, 35), (1323, 199), (1288, 26), (1269, 197), (1224, 17), (1218, 183), (1157, 23), 
            (1155, 191), (1082, 28), (1106, 197), (1026, 29), (1041, 203), (951, 51), (996, 206), 
            (878, 83), (981, 227), (834, 151), (962, 262), (812, 207), (931, 302), (783, 281), 
            (909, 379), (752, 320), (835, 467), (700, 345), (744, 503), (666, 346), (657, 524), 
            (608, 351), (597, 527), (531, 351), (516, 524), (459, 353), (459, 520), (366, 367), 
            (411, 533), (294, 382), (387, 548), (236, 423), (342, 584), (192, 469), (328, 587), 
            (162, 530), (315, 599), (147, 599), (303, 633), (142, 675), (301, 662), (156, 740), 
            (311, 695), (177, 801), (326, 731), (213, 879), (353, 755), (276, 903), (373, 777), 
            (339, 935), (408, 785), (402, 945), (435, 797), (487, 956), (480, 805), (558, 961), 
            (547, 803)
        ]

        self.track = []
        for i in range(len(self.track_outer_cor)-1):
            wall = Boundary(self.track_outer_cor[i][0], self.track_outer_cor[i][1],
                            self.track_outer_cor[i+1][0], self.track_outer_cor[i+1][1])
            self.track.append(wall)

        for i in range(len(self.track_inner_cor)-1):
            wall = Boundary(self.track_inner_cor[i][0], self.track_inner_cor[i][1],
                            self.track_inner_cor[i+1][0], self.track_inner_cor[i+1][1])
            self.track.append(wall)

        self.reward_gates = []
        for i in range(0, len(self.reward_gates_cor)-1, 2):
            gate = RewardGate(self.reward_gates_cor[i][0], self.reward_gates_cor[i][1],
                            self.reward_gates_cor[i+1][0], self.reward_gates_cor[i+1][1])
            self.reward_gates.append(gate)

        self.total_reward_gates = len(self.reward_gates)

        # Assuming self.width and self.height are the track's width and height, respectively
        # Update the minimum and maximum bounds for the observation space
        low_obs = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0])
        high_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Assuming you have num_rays distances from the car to the walls
        for _ in range(self.car.num_rays):
            low_obs = np.append(low_obs, 0.0)  # Add the minimum distance value (0.0) for each ray
            high_obs = np.append(high_obs, 1.0)  # Add the maximum distance value (1.0) for each ray

        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def _get_obs(self):
        obs = []
        obs.append(self.car.pos[0] / self.width)
        obs.append(self.car.pos[1] / self.height)
        obs.append(self.car.velocity[0] / self.width)
        obs.append(self.car.velocity[1] / self.height)

        x, y = np.cos(np.radians(self.car.rotation)), np.sin(np.radians(self.car.rotation))
        obs.append(x)
        obs.append(y)

        distances = self.car.get_distances(self.track)
        for distance in distances:
            obs.append(distance / self.width)

        obs = np.array(obs)

        return obs


    def _get_info(self):
        info = {}
        info['gates_passed'] = RewardGate.gates_passed
        info['time_passed'] = self.time_passed
        return info


    def reset(self, seed=None, options=None):
        self.time_passed = 0
        self.car.reset()
        self.car.destroyed = False
        RewardGate.gates_passed = 0
        RewardGate.gates_left = self.total_reward_gates
        for gate in self.reward_gates:
            gate.set_active()
        if options == "no_time_limit":
            self.max_time = np.inf
        else:
            self.max_time = 1000
        observation = self._get_obs()
        info = self._get_info()
        return observation
    

    def step(self, action):
        reward = 0
        # action translation
        if action == 1:
            self.car.move_car('forward')
            reward += 0.01
        elif action == 2:
            self.car.move_car('backward')
        elif action == 3:
            self.car.move_car('left')
        elif action == 4:
            self.car.move_car('right')
        elif action == 5:
            self.car.move_car('forward')
            self.car.move_car('left')
            reward += 0.01
        elif action == 6:
            self.car.move_car('forward')
            self.car.move_car('right')
            reward += 0.01
        elif action == 7:
            self.car.move_car('backward')
            self.car.move_car('left')
        elif action == 8:
            self.car.move_car('backward')
            self.car.move_car('right')

        # update
        if RewardGate.gates_left == 0:
            reward += 10
            for gate in self.reward_gates:
                gate.set_active()
            RewardGate.gates_left = self.total_reward_gates
        if self.car.check_gate_pass(self.reward_gates):
            reward += 1
        else:
            reward += 0
        self.car.update(self.track)
        self.time_passed += 1

        # check if terminated
        if self.car.destroyed:
            terminated = True
            reward -= 2
        elif self.time_passed > self.max_time:
            terminated = True
        else:
            terminated = False

        # get observation, reward, info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.width, self.height)
            )

            # Load the background and car images
            if self.background is None:
                self.background = pygame.image.load("C:\\Users\\Florian\\Documents\\GitHub\\car-driving-agent\\example_custome_environment\\track.png")
                self.background = pygame.transform.scale(self.background, (self.width, self.height))
            if self.car.image is None:
                self.car.set_image("C:\\Users\\Florian\\Documents\\GitHub\\car-driving-agent\\example_custome_environment\\car.png")
                
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0,0,0))
        canvas.blit(self.background, (0,0))
        # for gate in self.reward_gates:
        #     gate.display(canvas)
        self.car.display(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()