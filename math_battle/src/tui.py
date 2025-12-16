"""Text User Interface for Math Battle.

A terminal-based game interface that allows humans to play against AI agents.
"""

import os
import sys
from typing import Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .game_state import (
    GameState, MAX_ABILITIES,
    ATTR_HEALTH, ATTR_MAX_HEALTH, ATTR_MANA, ATTR_MAX_MANA,
    ATTR_STRENGTH, ATTR_DEFENSE, ATTR_BURN, ATTR_STUN,
    ATTRIBUTE_NAMES,
)
from .env import MathBattleFuncEnv, EnvParams
from .engine import get_action_mask
from .heroes import (
    create_fighter, create_fire_mage, HERO_NAMES, ABILITY_NAMES,
    get_ability_name,
)
from .agent import Agent, DummyAgent, RandomAgent, GreedyAgent


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_health_bar(current: float, maximum: float, width: int = 20) -> str:
    """Create a visual health bar."""
    if maximum <= 0:
        return "[" + " " * width + "]"

    ratio = max(0, min(1, current / maximum))
    filled = int(ratio * width)
    empty = width - filled

    if ratio > 0.6:
        color = Colors.GREEN
    elif ratio > 0.3:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    bar = f"{color}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"
    return f"[{bar}]"


def get_mana_bar(current: float, maximum: float, width: int = 15) -> str:
    """Create a visual mana bar."""
    if maximum <= 0:
        return ""

    ratio = max(0, min(1, current / maximum))
    filled = int(ratio * width)
    empty = width - filled

    bar = f"{Colors.BLUE}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"
    return f"[{bar}]"


class TUIEnvWrapper:
    """Wraps MathBattleFuncEnv to provide the API expected by the TUI.

    This effectively acts as the Gymnasium 'FunctionalJaxEnv' wrapper but
    tailored to the TUI's manual RNG handling preference.
    """
    def __init__(self, env: MathBattleFuncEnv, params: EnvParams):
        self.env = env
        self.params = params

    def reset(self, rng: jnp.ndarray) -> Tuple[GameState, jnp.ndarray, dict]:
        state = self.env.initial(rng, self.params)
        obs = self.env.observation(state, self.params)
        info = self._get_info(state)
        return state, obs, info

    def step(self, state: GameState, action: int, rng: jnp.ndarray):
        # Transition
        next_state = self.env.transition(state, action, rng, self.params)

        # Calculate derived values
        reward = self.env.reward(state, action, next_state, self.params)
        terminated = self.env.terminal(next_state, self.params)

        # Handle truncation check logic
        truncated = False
        if next_state.turn_count >= self.params.max_turns:
            truncated = True

        obs = self.env.observation(next_state, self.params)
        info = self._get_info(next_state)

        return next_state, obs, reward, terminated, truncated, info

    def _get_info(self, state: GameState) -> dict:
        return {
            "action_mask": np.array(get_action_mask(state)),
            "active_player": int(state.active_player),
            "turn_count": int(state.turn_count),
            "done": bool(state.done),
            "winner": int(state.winner),
        }


class MathBattleTUI:
    """Terminal User Interface for Math Battle."""

    def __init__(
        self,
        player_hero: str = "fighter",
        opponent_hero: str = "fire_mage",
        opponent_agent: Optional[Agent] = None,
        seed: int = 42,
    ):
        """Initialize the TUI.

        Args:
            player_hero: Hero type for player ("fighter" or "fire_mage")
            opponent_hero: Hero type for opponent
            opponent_agent: AI agent for opponent (default: DummyAgent)
            seed: Random seed
        """
        self.player_hero = player_hero
        self.opponent_hero = opponent_hero

        # Create heroes
        player_entity = HERO_NAMES[player_hero]()
        opponent_entity = HERO_NAMES[opponent_hero]()

        # Create environment and params
        base_env = MathBattleFuncEnv()
        params = EnvParams(
            player_template=player_entity,
            opponent_template=opponent_entity,
            dense_reward=False,
            max_turns=100,
        )
        # Use wrapper to maintain TUI compatibility
        self.env = TUIEnvWrapper(base_env, params)

        # Set up opponent agent
        if opponent_agent is None:
            self.opponent_agent = DummyAgent(hero_type=opponent_hero)
        else:
            self.opponent_agent = opponent_agent

        # Random key
        self.rng = jax.random.PRNGKey(seed)

        # Game state
        self.state: Optional[GameState] = None
        self.game_log: list = []

    def _split_rng(self):
        """Split the random key into two."""
        self.rng, new_key = jax.random.split(self.rng)
        return self.rng, new_key

    def start_game(self):
        """Start a new game."""
        self.rng, reset_rng = self._split_rng()
        self.state, obs, info = self.env.reset(reset_rng)
        self.game_log = []
        self.log_message("Game started!")
        self.log_message(f"You are playing as {Colors.CYAN}{self.player_hero.replace('_', ' ').title()}{Colors.RESET}")
        self.log_message(f"Opponent is {Colors.MAGENTA}{self.opponent_hero.replace('_', ' ').title()}{Colors.RESET}")
        self.log_message("")

    def log_message(self, message: str):
        """Add a message to the game log."""
        self.game_log.append(message)
        # Keep only last 10 messages
        if len(self.game_log) > 10:
            self.game_log = self.game_log[-10:]

    def render_entity(
        self,
        name: str,
        hero_type: str,
        attributes: np.ndarray,
        is_active: bool,
        is_player: bool,
    ) -> str:
        """Render entity status."""
        lines = []

        # Header
        active_marker = f"{Colors.GREEN}>{Colors.RESET} " if is_active else "  "
        color = Colors.CYAN if is_player else Colors.MAGENTA
        lines.append(f"{active_marker}{color}{Colors.BOLD}{name}{Colors.RESET}")

        # Health bar
        health = float(attributes[ATTR_HEALTH])
        max_health = float(attributes[ATTR_MAX_HEALTH])
        health_bar = get_health_bar(health, max_health)
        lines.append(f"  HP: {health_bar} {int(health)}/{int(max_health)}")

        # Mana bar (if applicable)
        mana = float(attributes[ATTR_MANA])
        max_mana = float(attributes[ATTR_MAX_MANA])
        if max_mana > 0:
            mana_bar = get_mana_bar(mana, max_mana)
            lines.append(f"  MP: {mana_bar} {int(mana)}/{int(max_mana)}")

        # Stats
        strength = int(attributes[ATTR_STRENGTH])
        defense = int(attributes[ATTR_DEFENSE])
        lines.append(f"  STR: {Colors.RED}{strength}{Colors.RESET}  DEF: {Colors.BLUE}{defense}{Colors.RESET}")

        # Status effects
        effects = []
        burn = int(attributes[ATTR_BURN])
        stun = int(attributes[ATTR_STUN])
        if burn > 0:
            effects.append(f"{Colors.RED}Burn({burn}){Colors.RESET}")
        if stun > 0:
            effects.append(f"{Colors.YELLOW}Stun({stun}){Colors.RESET}")
        if effects:
            lines.append(f"  Status: {' '.join(effects)}")

        return "\n".join(lines)

    def render_abilities(self, hero_type: str, action_mask: np.ndarray) -> str:
        """Render available abilities."""
        lines = [f"\n{Colors.BOLD}Your Abilities:{Colors.RESET}"]

        ability_names = ABILITY_NAMES.get(hero_type, [])
        for i, name in enumerate(ability_names):
            if i < len(action_mask) and action_mask[i]:
                lines.append(f"  [{i + 1}] {name}")
            elif i < len(action_mask):
                lines.append(f"  {Colors.DIM}[{i + 1}] {name} (unavailable){Colors.RESET}")

        return "\n".join(lines)

    def render_game_log(self) -> str:
        """Render the game log."""
        lines = [f"\n{Colors.BOLD}Game Log:{Colors.RESET}"]
        for msg in self.game_log[-6:]:
            lines.append(f"  {msg}")
        return "\n".join(lines)

    def render(self):
        """Render the full game screen."""
        clear_screen()

        if self.state is None:
            print("No game in progress. Call start_game() first.")
            return

        print(f"\n{'=' * 50}")
        print(f"{Colors.BOLD}         MATH BATTLE{Colors.RESET}")
        print(f"{'=' * 50}\n")

        # Get attributes
        player_attrs = np.array(self.state.player.attributes)
        opponent_attrs = np.array(self.state.opponent.attributes)
        active_player = int(self.state.active_player)
        turn = int(self.state.turn_count)

        print(f"Turn: {turn}")
        print(f"{'-' * 50}\n")

        # Render opponent
        print(self.render_entity(
            f"Opponent ({self.opponent_hero.replace('_', ' ').title()})",
            self.opponent_hero,
            opponent_attrs,
            is_active=(active_player == 1),
            is_player=False,
        ))

        print(f"\n{'-' * 50}\n")

        # Render player
        print(self.render_entity(
            f"You ({self.player_hero.replace('_', ' ').title()})",
            self.player_hero,
            player_attrs,
            is_active=(active_player == 0),
            is_player=True,
        ))

        # Render abilities (only if player's turn)
        if active_player == 0:
            action_mask = np.array(get_action_mask(self.state))
            print(self.render_abilities(self.player_hero, action_mask))

        # Render game log
        print(self.render_game_log())

        print(f"\n{'=' * 50}")

    def get_player_action(self) -> int:
        """Get action from human player."""
        action_mask = np.array(get_action_mask(self.state))
        ability_names = ABILITY_NAMES.get(self.player_hero, [])
        valid_actions = np.where(action_mask)[0]

        while True:
            try:
                choice = input("\nChoose ability (number) or 'q' to quit: ").strip().lower()

                if choice == 'q':
                    return -1  # Signal to quit

                action = int(choice) - 1  # Convert to 0-indexed

                if action in valid_actions:
                    return action
                else:
                    print(f"{Colors.RED}Invalid choice. Please select a valid ability.{Colors.RESET}")

            except ValueError:
                print(f"{Colors.RED}Please enter a number.{Colors.RESET}")
            except KeyboardInterrupt:
                return -1

    def execute_player_turn(self, action: int):
        """Execute the player's turn."""
        ability_name = get_ability_name(self.player_hero, action)
        self.log_message(f"You used {Colors.CYAN}{ability_name}{Colors.RESET}!")

        self.rng, step_rng = self._split_rng()
        self.state, obs, reward, terminated, truncated, info = self.env.step(
            self.state, action, step_rng
        )

    def execute_opponent_turn(self):
        """Execute the opponent's turn."""
        self.rng, agent_rng = self._split_rng()
        action = self.opponent_agent.select_action(self.state, agent_rng)

        ability_name = get_ability_name(self.opponent_hero, action)
        self.log_message(f"Opponent used {Colors.MAGENTA}{ability_name}{Colors.RESET}!")

        self.rng, step_rng = self._split_rng()
        self.state, obs, reward, terminated, truncated, info = self.env.step(
            self.state, action, step_rng
        )

    def check_game_over(self) -> Tuple[bool, Optional[str]]:
        """Check if game is over and return winner message."""
        if self.state.done:
            winner = int(self.state.winner)
            if winner == 0:
                return True, f"{Colors.GREEN}{Colors.BOLD}YOU WIN!{Colors.RESET}"
            elif winner == 1:
                return True, f"{Colors.RED}{Colors.BOLD}YOU LOSE!{Colors.RESET}"
            else:
                return True, f"{Colors.YELLOW}{Colors.BOLD}DRAW!{Colors.RESET}"
        return False, None

    def run(self):
        """Run the main game loop."""
        self.start_game()

        while True:
            self.render()

            # Check for game over
            game_over, result_msg = self.check_game_over()
            if game_over:
                print(f"\n{result_msg}\n")
                play_again = input("Play again? (y/n): ").strip().lower()
                if play_again == 'y':
                    self.start_game()
                    continue
                else:
                    print("Thanks for playing!")
                    break

            active_player = int(self.state.active_player)

            if active_player == 0:
                # Player's turn
                action = self.get_player_action()
                if action < 0:  # Quit
                    print("\nThanks for playing!")
                    break
                self.execute_player_turn(action)
            else:
                # Opponent's turn
                print(f"\n{Colors.DIM}Opponent is thinking...{Colors.RESET}")
                import time
                time.sleep(0.5)  # Small delay for better UX
                self.execute_opponent_turn()


def select_hero() -> str:
    """Let user select their hero."""
    print(f"\n{Colors.BOLD}Choose your hero:{Colors.RESET}")
    print(f"  [1] {Colors.CYAN}Fighter{Colors.RESET} - High HP, physical attacks")
    print(f"  [2] {Colors.MAGENTA}Fire Mage{Colors.RESET} - Spells, burn, healing")

    while True:
        choice = input("\nYour choice (1-2): ").strip()
        if choice == "1":
            return "fighter"
        elif choice == "2":
            return "fire_mage"
        print("Invalid choice. Please enter 1 or 2.")


def select_opponent() -> Tuple[str, Agent]:
    """Let user select opponent hero and agent."""
    print(f"\n{Colors.BOLD}Choose opponent hero:{Colors.RESET}")
    print(f"  [1] {Colors.CYAN}Fighter{Colors.RESET}")
    print(f"  [2] {Colors.MAGENTA}Fire Mage{Colors.RESET}")

    while True:
        choice = input("\nOpponent hero (1-2): ").strip()
        if choice == "1":
            hero = "fighter"
            break
        elif choice == "2":
            hero = "fire_mage"
            break
        print("Invalid choice.")

    print(f"\n{Colors.BOLD}Choose opponent difficulty:{Colors.RESET}")
    print("  [1] Random - picks random abilities")
    print("  [2] Greedy - always picks highest damage")
    print("  [3] Smart  - uses heuristics")

    while True:
        choice = input("\nDifficulty (1-3): ").strip()
        if choice == "1":
            return hero, RandomAgent()
        elif choice == "2":
            return hero, GreedyAgent(hero_type=hero)
        elif choice == "3":
            return hero, DummyAgent(hero_type=hero)
        print("Invalid choice.")


def main():
    """Main entry point for TUI."""
    clear_screen()
    print(f"\n{'=' * 50}")
    print(f"{Colors.BOLD}       WELCOME TO MATH BATTLE{Colors.RESET}")
    print(f"{'=' * 50}")
    print("\nA turn-based fantasy duel game!")

    # Hero selection
    player_hero = select_hero()
    opponent_hero, opponent_agent = select_opponent()

    # Create and run TUI
    tui = MathBattleTUI(
        player_hero=player_hero,
        opponent_hero=opponent_hero,
        opponent_agent=opponent_agent,
    )

    try:
        tui.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")


if __name__ == "__main__":
    main()
