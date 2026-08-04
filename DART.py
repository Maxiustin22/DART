import pygame
import math
import random
import numpy as np
import os


BAZA_X = 250
BAZA_Y = 100  # Am pus baza sus ca sa aiba spatiu sa zboare

MOD_TEST = False
SEEDURI_TEST = list(range(100, 120))
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)


class Creier:
    def __init__(self):
        self.W1 = np.random.randn(4, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32) * 0.1
        self.b2 = 0.0

    def copiaza_si_muteaza(self, rata_mutatie=0.05):
        copil = Creier()
        copil.W1 = np.copy(self.W1)
        copil.b1 = np.copy(self.b1)
        copil.W2 = np.copy(self.W2)
        copil.b2 = self.b2

        copil.W1 += np.random.randn(*self.W1.shape) * rata_mutatie
        copil.b1 += np.random.randn(*self.b1.shape) * rata_mutatie
        copil.W2 += np.random.randn(*self.W2.shape) * rata_mutatie
        copil.b2 += np.random.randn() * rata_mutatie
        return copil

    def salveaza(self, nume_fisiere="creier_salvat.npz"):
        np.savez(nume_fisiere, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def incarca(self,nume_fisiere="creier_salvat.npz"):
        date = np.load(nume_fisiere)
        self.W1 = date["W1"]
        self.b1 = date["b1"]
        self.W2 = date["W2"]
        self.b2 = date["b2"]

class Racheta:
    def __init__(self, spawn_x, spawn_y, viteza):
        self.x_spawn = spawn_x
        self.y_spawn = spawn_y
        self.viteza = viteza
        self.vx = 0
        self.vy = 0
        self.inaltime = 10
        self.latime = 5
        self.culoare = (255, 10, 10)

    def seteaza_tinta(self, tx, ty):
        dx = tx - self.x_spawn
        dy = ty - self.y_spawn
        distanta = math.hypot(dx, dy)
        if distanta > 0:
            self.vx = (dx / distanta) * self.viteza
            self.vy = (dy / distanta) * self.viteza

    def misca(self, dt):
        self.x_spawn += self.vx * dt
        self.y_spawn += self.vy * dt

    def deseneaza(self, win):
        dreptunghi = (int(self.x_spawn), int(self.y_spawn), self.latime, self.inaltime)
        pygame.draw.rect(win, self.culoare, dreptunghi)


class rachetaInamica:
    def __init__(self, x, y, viteza, creier_primit):
        self.x_spawn = x
        self.y_spawn = y
        self.viteza = viteza
        self.inaltime = 10
        self.latime = 10
        self.culoare = (88, 100, 200)

        dx = BAZA_X - self.x_spawn
        dy = BAZA_Y - self.y_spawn
        self.unghi = math.atan2(dy, dx)

        self.creier = creier_primit
        self.deschidere = math.pi / 4

        self.timp_viata = 0.0
        self.timp_maxim = 25.0

        self.moarta = False
        self.fitness = 0.0

        self.best_dist = math.hypot(x - BAZA_X, y - BAZA_Y)
        self.start_dist = math.hypot(x - BAZA_X, y - BAZA_Y)

    def gandeste_si_misca(self, dt, obstacole):
        if self.moarta:
            return

        self.timp_viata += dt

        dist_stanga = 50.0
        dist_centru = 50.0
        dist_dreapta = 50.0

        for obs in obstacole:
            dist = math.hypot(obs[0] - self.x_spawn, obs[1] - self.y_spawn)
            if dist < 50.0:
                unghi_obs = math.atan2(obs[1] - self.y_spawn, obs[0] - self.x_spawn)
                dif = (unghi_obs - self.unghi + math.pi) % (2 * math.pi) - math.pi

                if -self.deschidere - 0.3 < dif < -self.deschidere + 0.3:
                    if dist < dist_stanga: dist_stanga = dist
                elif -0.3 < dif < 0.3:
                    if dist < dist_centru: dist_centru = dist
                elif self.deschidere - 0.3 < dif < self.deschidere + 0.3:
                    if dist < dist_dreapta: dist_dreapta = dist

        val_stanga = 1.0 - (dist_stanga / 50.0)
        val_centru = 1.0 - (dist_centru / 50.0)
        val_dreapta = 1.0 - (dist_dreapta / 50.0)


        unghi_baza = math.atan2(BAZA_Y - self.y_spawn, BAZA_X - self.x_spawn)
        dif_baza = (unghi_baza - self.unghi + math.pi) % (2 * math.pi) - math.pi
        baza_semnal = dif_baza / math.pi

        inputs = np.array([val_stanga, val_centru, val_dreapta, baza_semnal])

        #hidden layers tanh and relu
        scor_ascuns = np.dot(inputs, self.creier.W1) + self.creier.b1
        hidden_layer = relu(scor_ascuns)

        scor_final = np.dot(hidden_layer, self.creier.W2) + self.creier.b2

        # Extragem din array cu [0]
        probabilitate_volan = tanh(scor_final)
        putere_volan = probabilitate_volan * 3.0
        self.unghi += putere_volan * dt

        self.x_spawn += math.cos(self.unghi) * self.viteza * dt
        self.y_spawn += math.sin(self.unghi) * self.viteza * dt

        d = math.hypot(self.x_spawn - BAZA_X, self.y_spawn - BAZA_Y)
        if d < self.best_dist:
            self.best_dist = d

    def deseneaza(self, win):
        if self.moarta:
            return

        dreptunghi = (int(self.x_spawn), int(self.y_spawn), self.latime, self.inaltime)
        pygame.draw.rect(win, self.culoare, dreptunghi)

        procent_timp = max(0, 1.0 - (self.timp_viata / self.timp_maxim))
        lungime_bara = self.latime * 2
        pygame.draw.rect(win, (255, 0, 0), (int(self.x_spawn) - 2, int(self.y_spawn) - 8, lungime_bara, 3))
        pygame.draw.rect(win, (0, 255, 0),
                         (int(self.x_spawn) - 2, int(self.y_spawn) - 8, int(lungime_bara * procent_timp), 3))

    def deseneaza_senzori(self, win):
        if self.moarta: return
        raza = 50.0
        unghiuri = [self.unghi - self.deschidere, self.unghi, self.unghi + self.deschidere]
        for u in unghiuri:
            ex = self.x_spawn + math.cos(u) * raza
            ey = self.y_spawn + math.sin(u) * raza
            pygame.draw.line(win, (0, 255, 0), (self.x_spawn, self.y_spawn), (ex, ey), 1)



win = pygame.display.set_mode((500, 500))
pygame.display.set_caption("DART - AI Evolutiv")
ceas = pygame.time.Clock()

racheta_mea = Racheta(250, 400, 15)
racheta_mea.seteaza_tinta(110, 200)

random.seed(42)

obstacole = []
for _ in range(20):
    obstacole.append((random.randint(50, 450), random.randint(150, 350)))

POPULATIE = 20

generatia = 1
#baza_mutata = False
best_fitness_istoric = -999999.0
with open("istoric_antrenament.txt", "w") as fisier_text:
    fisier_text.write("JURNAL\n")
    fisier_text.write("Generatia | Scor Fitness | Timp \n")
    fisier_text.write("-" * 50 + "\n")
ruleaza = True

if MOD_TEST:
    index_seed_test = 0
    rezultate_test = []
    with open("rezultate_test.txt", "w") as ft:
        ft.write("TEST ZERO-SHOT — elita salvata, harti nevazute\n")
        ft.write("Seed | Atins baza | Fitness | Timp\n")
        ft.write("-" * 45 + "\n")

def genereaza_harta(seed_harta):
    random.seed(seed_harta)
    obstacole.clear()
    for _ in range(20):
        obstacole.append((random.randint(50, 450), random.randint(150, 350)))

def spawn_valid(obstacole):
    for i in range(100):
        x_random_spawn = random.randint(20, 480)
        y_random_spawn = random.randint(20, 480)
        if math.hypot(x_random_spawn - BAZA_X, y_random_spawn - BAZA_Y) < 150:
            continue

        too_close = False
        for obs in obstacole:
            if math.hypot(obs[0] - x_random_spawn, obs[1] - y_random_spawn) < 50:
                too_close = True
                break

        if not too_close:
            return x_random_spawn, y_random_spawn

    return 250, 450



elita_incarcata = []
if os.path.exists("creier_salvat.npz"):
    date = np.load("creier_salvat.npz")
    if "W1_0" in date:
        for k in range(3):
            c = Creier()
            c.W1 = date[f"W1_{k}"]; c.b1 = date[f"b1_{k}"]
            c.W2 = date[f"W2_{k}"]; c.b2 = float(date[f"b2_{k}"])
            elita_incarcata.append(c)
        print("Elita (3 creiere) incarcata cu succes.")
    else:
        c = Creier()
        c.incarca("creier_salvat.npz")
        elita_incarcata.append(c)
        print("Creier (format vechi) incarcat cu succes.")

roi_inamici = []
if elita_incarcata:
    for c in elita_incarcata:
        roi_inamici.append(rachetaInamica(250, 450, 20, c))
    for _ in range(POPULATIE - len(elita_incarcata)):
        parinte = random.choice(elita_incarcata)
        roi_inamici.append(rachetaInamica(250, 450, 20, parinte.copiaza_si_muteaza(rata_mutatie=0.05)))
else:
    for _ in range(POPULATIE):
        roi_inamici.append(rachetaInamica(250, 450, 20, Creier()))

if MOD_TEST:
    if not elita_incarcata:
        print("EROARE: MOD_TEST cere un creier_salvat.npz existent (elita antrenata)!")
        ruleaza = False
    else:
        genereaza_harta(SEEDURI_TEST[0])
        print(f"TEST: harta seed {SEEDURI_TEST[0]}")

while ruleaza:
    ceas.tick(60)
    dt = 1.0 / 60.0

    if not pygame.display.get_init():
        break

    for eveniment in pygame.event.get():
        if eveniment.type == pygame.QUIT:
            ruleaza = False

    win.fill((0, 0, 0))

    # Desenam baza
    pygame.draw.rect(win, (80, 50, 25), (BAZA_X - 7, BAZA_Y - 7, 15, 15), 15)

    racheta_mea.misca(dt)
    racheta_mea.deseneaza(win)
    if racheta_mea.x_spawn < 0 or racheta_mea.x_spawn > 500 or racheta_mea.y_spawn < 0 or racheta_mea.y_spawn > 500:
        racheta_mea = Racheta(250, 400, 15)
        racheta_mea.seteaza_tinta(110, 200)

    for obs in obstacole:
        pygame.draw.circle(win, (255, 255, 0), obs, 8)

    toate_moarte = True

    for i, inamic in enumerate(roi_inamici):
        if not inamic.moarta:
            toate_moarte = False
            inamic.gandeste_si_misca(dt, obstacole)
            inamic.deseneaza(win)

            if i == 0:
                inamic.deseneaza_senzori(win)

            lovit = False
            for obs in obstacole:
                if math.hypot(inamic.x_spawn - obs[0], inamic.y_spawn - obs[1]) < 12:
                    lovit = True

            if inamic.x_spawn < 0 or inamic.x_spawn > 500 or inamic.y_spawn < 0 or inamic.y_spawn > 500:
                lovit = True

            # 2. CALCUL FITNESS
            distanta_baza = math.hypot(inamic.x_spawn - BAZA_X, inamic.y_spawn - BAZA_Y)

            if distanta_baza < 20:
                inamic.fitness = 10000 + (inamic.timp_maxim - inamic.timp_viata) * 100
                inamic.moarta = True
            elif lovit or inamic.timp_viata >= inamic.timp_maxim:
                inamic.progres = (inamic.start_dist - inamic.best_dist) / inamic.start_dist
                inamic.fitness = inamic.progres * 1000
                inamic.fitness -= inamic.timp_viata * 15
                if inamic.timp_viata >= inamic.timp_maxim:
                    inamic.fitness -= 300
                inamic.moarta = True

    if toate_moarte and MOD_TEST:
        roi_inamici.sort(key=lambda x: x.fitness, reverse=True)
        campion = roi_inamici[0]
        atins = campion.fitness >= 10000
        seed_curent = SEEDURI_TEST[index_seed_test]
        rezultate_test.append(atins)
        with open("rezultate_test.txt", "a") as ft:
            ft.write(f"{seed_curent:4d} | {'DA ' if atins else 'NU '} | {campion.fitness:8.1f} | {campion.timp_viata:4.1f} sec\n")
        print(f"Seed {seed_curent}: {'ATINS' if atins else 'ratat'} ({campion.fitness:.1f})")

        index_seed_test += 1
        if index_seed_test >= len(SEEDURI_TEST):
            procent = 100.0 * sum(rezultate_test) / len(rezultate_test)
            with open("rezultate_test.txt", "a") as ft:
                ft.write("-" * 45 + "\n")
                ft.write(f"REZULTAT: {sum(rezultate_test)}/{len(rezultate_test)} harti atinse = {procent:.0f}%\n")
            print(f"=== TEST INCHEIAT: {procent:.0f}% zero-shot ===")
            ruleaza = False
        else:
            genereaza_harta(SEEDURI_TEST[index_seed_test])
            roi_inamici.clear()
            for c in elita_incarcata:
                roi_inamici.append(rachetaInamica(250, 450, 20, c))
            for _ in range(POPULATIE - len(elita_incarcata)):
                parinte = random.choice(elita_incarcata)
                roi_inamici.append(rachetaInamica(250, 450, 20, parinte.copiaza_si_muteaza(rata_mutatie=0.05)))

    elif toate_moarte:
        roi_inamici.sort(key=lambda x: x.fitness, reverse=True)
        campion = roi_inamici[0]

        print(f"Generatia {generatia} s-a incheiat | Cel mai bun scor: {campion.fitness:.1f}")

        with open("istoric_antrenament.txt", "a") as fisier_text:
            fisier_text.write(f"Gen: {generatia:03d} | Fitness: {campion.fitness:8.1f} | Timp: {campion.timp_viata:4.1f} sec\n")

        elita = [roi_inamici[0].creier, roi_inamici[1].creier, roi_inamici[2].creier]
        if campion.fitness > best_fitness_istoric:
            best_fitness_istoric = campion.fitness
            np.savez("creier_salvat.npz",
                     W1_0=elita[0].W1, b1_0=elita[0].b1, W2_0=elita[0].W2, b2_0=np.array(elita[0].b2),
                     W1_1=elita[1].W1, b1_1=elita[1].b1, W2_1=elita[1].W2, b2_1=np.array(elita[1].b2),
                     W1_2=elita[2].W1, b1_2=elita[2].b1, W2_2=elita[2].W2, b2_2=np.array(elita[2].b2))
            print(f"  -> RECORD NOU salvat pe disc: {campion.fitness:.1f}")

        roi_inamici.clear()

        for creier_elita in elita:
            roi_inamici.append(rachetaInamica(250, 450, 20, creier_elita))

        for _ in range(POPULATIE - 3):
            parinte = random.choice(elita)
            creier_mutant = parinte.copiaza_si_muteaza(rata_mutatie=0.05)
            roi_inamici.append(rachetaInamica(250, 450, 20, creier_mutant))

        generatia += 1

#   elif eveniment.type == pygame.MOUSEBUTTONDOWN and not baza_mutata:
#       BAZA_X, BAZA_Y = eveniment.pos
#      baza_mutata = True

    pygame.display.flip()

pygame.quit()
