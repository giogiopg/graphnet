import sqlite3
import numpy as np
from weight_events_oscprob import compute_evt_weight
import sys


def do_weights(db_path):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch the necessary columns from the 'truth' table
    cursor.execute('''
        SELECT event_no, pdgid, Energy, part_dir_z, is_cc_flag, w2, n_gen, livetime
        FROM truth
    ''')

    rows = cursor.fetchall()

    # Extract columns into numpy arrays for computation
    pdgid = np.array([row[1] for row in rows])
    Energy = np.array([row[2] for row in rows])
    dir_z = np.array([row[3] for row in rows])
    is_cc_flag = np.array([row[4] for row in rows])
    w2 = np.array([row[5] for row in rows])
    n_gen = np.array([row[6] for row in rows])
    livetime = np.array([row[7] for row in rows])

    w_osc = compute_evt_weight(pdgid, Energy, dir_z, is_cc_flag, w2, livetime, n_gen)

    update_data = [(w_osc[i], rows[i][0]) for i in range(len(rows))]

    cursor.executemany('''
        UPDATE truth
        SET w_osc = ?
        WHERE event_no = ?
    ''', update_data)

    conn.commit()
    conn.close()
    print(f"Done updating the weights of file {db_path}")

if __name__=="__main__":
    do_weights(sys.argv[1])
