"""
Package de registres centralisés du projet.

Ce package regroupe les dictionnaires de configuration partagés entre plusieurs points d'entrée (`play_game.py`, `step_by_step_run.py`,
`research.run_simulation`, etc.), afin qu'un nouvel agent ou une nouvelle option ne nécessite qu'une seule modification centralisée plutôt
qu'une répétition manuelle dans chaque script consommateur.
"""
