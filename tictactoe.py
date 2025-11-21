import tkinter as tk
from etat import Etat
import json
import itertools
import random
import numpy as np

window = tk.Tk()
window.title("Tic Tac Toe")
window.tour_x = True

# ================================
#       Interface graphique
# ================================
buttons = [[tk.Button(window, font=('Arial', 20), width=5, height=2,
                     command=lambda i=i, j=j: placer(i, j))
            for j in range(3)] for i in range(3)]
for i, row in enumerate(buttons):
    for j, btn in enumerate(row):
        btn.grid(row=i, column=j, padx=5, pady=5)

# Bouton Rejouer
def rejouer():
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text="", state='normal')
    window.tour_x = True

btn_rejouer = tk.Button(window, text="Rejouer", font=('Arial', 14), bg='lightgreen', command=rejouer)
btn_rejouer.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=10)


# ================================
#       Logique du jeu
# ================================
def placer(i, j):
    if buttons[i][j]['text'] == "":
        buttons[i][j]['text'] = "X"
        window.tour_x = False
        if not verifier_fin():
            window.after(100, jouer_robot)

def jouer_robot():
    plateau = [buttons[i][j]['text'] or "" for i in range(3) for j in range(3)]
    etat_actuel = next((e for e in etats if e.plateau == plateau), None)
    if etat_actuel and not etat_actuel.est_final():
        action = pi[etat_actuel.id - 1] if etat_actuel.id and pi[etat_actuel.id - 1] != 0 else None
        if action is not None:
            pos = action - 1
            i, j = divmod(pos, 3)
            if buttons[i][j]['text'] == "":
                buttons[i][j]['text'] = "O"
                window.tour_x = True
                verifier_fin()
                return
        libres = [idx for idx, val in enumerate(plateau) if val == ""]
        if libres:
            pos = random.choice(libres)
            i, j = divmod(pos, 3)
            buttons[i][j]['text'] = "O"
            window.tour_x = True
            verifier_fin()

def verifier_fin():
    plateau = [buttons[i][j]['text'] or "" for i in range(3) for j in range(3)]
    e = Etat(); e.plateau = plateau
    if e.est_final():
        g = e.gagnant()
        message = "Match nul !"
        if g == "X":
            message = "Vous avez gagné !"
        elif g == "O":
            message = "Le robot a gagné !"
        print(message)
        for row in buttons:
            for btn in row:
                btn.config(state='disabled')
        return True
    return False


##################################################################################################
#                                   Création de états                                            #
##################################################################################################
symbols = ["X", "O", ""]

def is_valid_state(grid):
    count_x, count_o = grid.count("X"), grid.count("O")
    # X commence -> count_x == count_o or count_x == count_o + 1
    if count_x not in [count_o, count_o + 1]:
        return False
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    x_wins = any(grid[a]==grid[b]==grid[c]=="X" for a,b,c in wins)
    o_wins = any(grid[a]==grid[b]==grid[c]=="O" for a,b,c in wins)
    # état invalide si les deux gagnent ou si un gagnant n'a pas le bon nombre de coups
    if x_wins and count_x != count_o + 1:
        return False
    if o_wins and count_x != count_o:
        return False
    return True

all_grids = itertools.product(symbols, repeat=9)
etats = [
    (lambda e: setattr(e, 'reward',
        -10 if (g:=e.gagnant()) == 'X' else
        10 if g == 'O' else 0
    ) or e)(
        (lambda e: setattr(e, 'plateau', list(grille)) or e)(Etat(id_etat=i))
    )
    for i, grille in enumerate(filter(is_valid_state, all_grids), 1)
]

#####################################################################################################
#                                          Etat suivant                                             # 
#####################################################################################################
def obtenir_indices_suivants(etat_courant, etats_list):
    """
    etat_courant : état APRES le coup de l'agent (déjà placé dans neighbors).
    Retourne la liste des indices dans etats_list correspondant aux états
    possibles APRÈS la réponse de l'adversaire. Si etat_courant est final,
    retourne l'indice de cet état final (transition certaine).
    """
    indices_suivants = []
    count_x = etat_courant.plateau.count('X')
    count_o = etat_courant.plateau.count('O')
    prochain_symbole = 'X' if count_x == count_o else 'O'

    for i in range(9):
        if etat_courant.plateau[i] == '':
            nouveau_plateau = etat_courant.plateau.copy()
            nouveau_plateau[i] = prochain_symbole

            for idx, etat in enumerate(etats_list):
                if etat.plateau == nouveau_plateau:
                    indices_suivants.append(idx)
                    break
    return indices_suivants

#####################################################################################################
#                                        Calcule V*                                                 #
#####################################################################################################

v = [0] * len(etats)
gamma = 0.90
epsilon = 0.00001  # Seuil de convergence
max_iterations = 1000

for iteration in range(max_iterations):
    delta = 0
    old_v = v.copy()  # Important: travailler sur une copie
    
    for state in etats:
        if state.est_final():
            continue

        action_values = []
        
        for action in state.action_legales():
            etat_apres_agent = state.neighbors(action)
            indices = obtenir_indices_suivants(etat_apres_agent, etats)       
            
            val = 0.0
            # Calculer la valeur attendue pour cette action
            p_trans = 1.0 / len(indices) if indices else 0  
            for i in indices:
                val += p_trans * (etats[i].reward + gamma * v[i])
            action_values.append(val)

        if action_values:  
            new_value = max(action_values)
            v[state.id - 1] = new_value
            
    delta = np.sum(np.abs(np.array(old_v) - np.array(v)))
    print(delta)
    
    
    if delta < epsilon:
        print(f"Convergence après {iteration + 1} itérations (VE)")
        break
else:
    print(f"Pas de convergence après {max_iterations} itérations (VE)")

#####################################################################################################
#                               Trouver la meilleure politique                                        #
#####################################################################################################

# Initialisation
pi = [0] * len(etats)  # Politique optimale 
v = [0] * len(etats)
gamma = 0.90
epsilon = 0.001
max_iterations = 1000

for iteration in range(max_iterations):
    delta = 0
    new_v = v.copy()  # Travailer sur une copie
    
    for state in etats:
        if state.est_final():
            continue

        action_values = []
        legal_actions = list(state.action_legales())
        
        for action in legal_actions:
            etat_apres_agent = state.neighbors(action)
            indices = obtenir_indices_suivants(etat_apres_agent, etats)
            
            val = 0.0
            # Probabilité uniforme sur les états suivants
            p_trans = 1.0 / len(indices) if indices else 0
            for i in indices:
                val += p_trans * (etats[i].reward + gamma * v[i])
            action_values.append((action, val))

        if action_values:
            # Trouver la meilleure action et sa valeur
            best_action, new_value = max(action_values, key=lambda x: x[1])
            
            # Mettre à jour la nouvelle valeur
            new_v[state.id - 1] = new_value
            pi[state.id - 1] = best_action
    
    # Calculer delta global APRÈS toutes les mises à jour (comme dans la première partie)
    delta = np.sum(np.abs(np.array(v) - np.array(new_v)))
    print(f"Itération {iteration + 1}: delta = {delta}")
    
    v = new_v  # Mettre à jour toutes les valeurs
    
    if delta < epsilon:
        print(f"Convergence après {iteration + 1} itérations (politique)")
        break
else:
    print(f"Pas de convergence après {max_iterations} itérations (politique)")

# Afficher la politique optimale pour quelques états
print("\nPolitique optimale (premiers états):")
for i in range(min(3000, len(etats))):
    print(f"État {etats[i].id}: Plateau {etats[i].plateau} → Action optimale: {pi[i]}")
politique_json = [{"etat": etats[i].plateau, "action": pi[i]} for i in range(len(etats))]
json.dump(politique_json, open('policy.json', 'w'), indent=2)

window.mainloop()
