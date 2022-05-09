import random
import math

class Enemy :

    def __init__(self, id: int):
        self.id = id
        self.symbol = "E"
        self.action_list = ["North", "South", "East", "West"]
        self.position = None
        self.verbose = False

    def choose_action(self, agent_position: tuple, surroundings: list):
        """
        Return the next action of the enemy, as defined by Appendix A of the paper, page 28-29

        Args:
            agent_position (tuple): Current position of the agent
            surroundings (list): [description]

        Returns:
            String: Next action of the enemy
        """

        # The enemy will not move with a 20% probability
        if random.random() < 0.2:
            if self.verbose:
                print("Enemy ", self.id, " at ", self.position, " - Action taken : Stay")
            return "Stay"

        # Probability of moving in a direction [p_north, p_south, p_east_, p_west]
        prob_action_list = [0, 0, 0, 0]

        angle_list = self.compute_angles(agent_position)

        for i in range(len(surroundings)):
            # Ignore adjacent cells containing an obstacle
            if surroundings[i] == "0":
                continue
            w_angle = (180 - abs(angle_list[i])) / 180

            # Compute Manhattan distance
            dist = abs(agent_position[0] - self.position[0]) + abs(agent_position[1] - self.position[1])

            if dist <= 4:
                t_dist = 15 - dist
            elif dist <= 15:
                t_dist = 9 - (dist / 2)
            else:
                t_dist = 1
            
            prob_action_list[i] = math.exp(0.33 * w_angle * t_dist)

        sum_proba = sum(prob_action_list)
        prob_action_list = [proba_i / sum_proba for proba_i in prob_action_list]
        sorted_prob_action_list = sorted(prob_action_list)
        draw = random.random()
        proba_sum = 0
        for i in range(len(sorted_prob_action_list)):
            proba_sum += sorted_prob_action_list[i]
            if draw < proba_sum:
                if self.verbose:
                    print("Enemy ", self.id, " at ", self.position, " - Proba of actions : ", prob_action_list)
                    print("Enemy ", self.id, " at ", self.position, " - Action taken : ", self.action_list[prob_action_list.index(sorted_prob_action_list[i])])
                return self.action_list[prob_action_list.index(sorted_prob_action_list[i])]
        
        print("Enemy choose_action() : no action returned, should not happen")


    def compute_angles(self, agent_position: tuple):
        """
        Compute the angle between the direction of each action A_i, and the direction
        from the enemy to the agent

        Args:
            agent_position (tuple): Current position of the agent

        Returns:
            Integer list: [a_north, a_south, a_east, a_west]
        """

        vec1 = [self.position[0] - agent_position[0], self.position[1] - agent_position[1]] # Vector pointing from agent position to enemy position

        angle_list = []
        vec2 = [1, 0] # North direction
        angle_list.append(self.findAngle(vec1, vec2))
        vec2 = [-1, 0] # South direction
        angle_list.append(self.findAngle(vec1, vec2))
        vec2 = [0, -1] # East direction
        angle_list.append(self.findAngle(vec1, vec2))
        vec2 = [0, 1] # West direction
        angle_list.append(self.findAngle(vec1, vec2))
        return angle_list
        
    def findAngle(self, vec1, vec2):
        """
        Compute the angle between two 2D vectors

        Args:
            vec1 (tuple<int>): Vector representing the direction between the considered enemy and the agent
            vec2 (tuple<int>): Vector representing the considered move direction

        Returns:
            [float]: Angle value between the two vectors
        """
        angle = math.acos((vec1[0] * vec2[0] + vec1[1] * vec2[1]) / (math.sqrt(vec1[0]**2 + vec1[1]**2) * math.sqrt(vec2[0]**2 + vec2[1]**2)))
        angle = (angle * 180) / math.pi
        return angle

    def get_id(self):
        return self.id

    def get_symbol(self):
        return self.symbol

    def set_position(self, new_position: tuple):
        self.position = new_position

    def get_position(self):
        return self.position