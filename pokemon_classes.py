import random

class Attack:
    """
    Wraps an attack type and calculates damage.
    Uses the attacker's actual 'attack' stat from the CSV.
    """
    BASE_POWER = 40

    def __init__(self, attack_type: str):
        self.attack_type = attack_type

    def damage(self, attacker, defender, type_multiplier: float = 1.0) -> float:
        """
        Damage formula:
            damage = BASE_POWER * (attacker.attack / defender.defense)
                     * type_multiplier * random_factor
        """
        random_factor = random.uniform(0.85, 1.0)
        raw = (self.BASE_POWER
               * (attacker.attack / max(defender.defense, 1))
               * type_multiplier
               * random_factor)
        return round(raw, 1)

    def __repr__(self):
        return f"Attack(type={self.attack_type})"
