from datetime import datetime
import pandas as pd

#https://vytapeni.tzb-info.cz/tabulky-a-vypocty/97-vypocet-doby-ohrevu-teple-vody
class Bojler:
    def __init__(self, capacity=100, wattage=2000):
                
                
        print("------------------------------------------------------\n")
        print('initializing of control...\n\tCapacity of Bojler = {}\n\t Wattage of bojler = {}\n'.format(capacity, wattage))
        print("------------------------------------------------------\n")

        self.bojler_heat_cap = capacity * 1.163
        self.real_wattage = wattage * 0.98
 

    def time_needed_to_heat_up(self, tmp_change):
        """
            time = (m * c) * d'(tmp) / P * effectivity_coef
        """
        return (self.bojler_heat_cap * tmp_change) / (self.real_wattage )


    def is_needed_to_heat(self, tmp_act, tmp_goal, time_to_consumption):
        tmp_change = tmp_goal - tmp_act
               
        if (tmp_change > 0) and (time_to_consumption <= self.time_needed_to_heat_up(tmp_change)):
            return True
        else:
            return False
if __name__ == '__main__':
    tmp_act = 30
    tmp_goal = 52
    time_to_consumption = 1
    b = Bojler(capacity = 80, wattage =2000)
    print(b.is_needed_to_heat(tmp_act, tmp_goal, time_to_consumption))