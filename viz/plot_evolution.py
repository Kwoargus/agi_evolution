import matplotlib.pyplot as plt
from db.connector import get_connection

def plot_fitness_history():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT generation_num, best_fitness, avg_fitness FROM agi_evolution.generations ORDER BY generation_num")
    rows = cur.fetchall()
    gens, best, avg = zip(*rows)
    plt.plot(gens, best, label='Best')
    plt.plot(gens, avg, label='Average')
    plt.xlabel('Поколение')
    plt.ylabel('Фитнес')
    plt.legend()
    plt.show()