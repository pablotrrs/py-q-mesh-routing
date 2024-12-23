from enum import Enum
from dataclasses import dataclass
import random
import time
import yaml
import os
from classes import Application

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

# Ecuación de Bellman:
# ΔQ_x(d, y) = α * (s + t - Q_x(d, y))
# Donde:
# - s: Tiempo de transmisión, calculado como la diferencia entre el timestamp guardado y el tiempo actual.
# - t: Tiempo estimado restante que el nodo calculó cuando eligió el vecino al que enviaría el paquete.
# - Q_x(d, y): Valor Q actual para el nodo actual (x) hacia el vecino (y).
# - α (alpha): Tasa de aprendizaje.
BELLMAN_EQ = lambda s, t, q_current: q_current + ALPHA * (s + t - q_current)

class NodeFunction(Enum):
    A = "A"
    B = "B"
    C = "C"

FUNCTION_SEQ = [NodeFunction.A, NodeFunction.B, NodeFunction.C]

@dataclass
class CallbackChainStep:
    previous_hop_node: int  # Next node in the callback chain
    next_hop_node: int  # Node which we have to calculate the Q-Value for
    send_timestamp: float  # Timestamp del momento en que se realizó el salto to next hop
    estimated_time: float  # Tiempo estimado para completar la secuencia según la Q-table

class QRoutingApplication(Application):
    def __init__(self, node):
        self.node = node
        self.q_table = {}
        self.assigned_function = None
        self.callback_stack = []

    def receive_packet(self, packet):
        print(f'[Node_ID={self.node.node_id}] Received packet {packet}')

        if packet.type == PacketType.PACKET_HOP:
            self.handle_packet_hop(packet)
        elif packet.type == PacketType.CALLBACK:
            self.handle_echo_callback(packet)

    def handle_packet_hop(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def handle_echo_callback(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_q_value(self, next_node, s, t):
        """
        Actualiza el valor Q para el nodo actual y la acción (saltar al vecino `next_node`) 
        usando la ecuación de Bellman.
        """
        old_q = self.q_table[self.node.node_id].get(next_node, 0.0)
        new_q = BELLMAN_EQ(s, t, old_q)

        self.q_table[self.node.node_id][next_node] = new_q

        print(
            f"[Node_ID={self.node.node_id}] Updated Q-Value for state {self.node.node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (estimated time {t}, actual time {s})"
        )

    def select_next_node(self) -> int:
        """
        Selecciona el siguiente nodo basado en la política ε-greedy con epsilon decay.
        """
        self.initialize_or_update_q_table()
        global EPSILON

        if random.random() < EPSILON:
            # Exploración: elegir un vecino aleatorio
            print(f'[Node_ID={self.node.node_id}] Performing exploration with epsilon={EPSILON:.4f}')
            next_node = random.choice(self.node.network.get_neighbors(self.node.node_id))
        else:
            # Explotación: elegir la mejor acción
            print(f'[Node_ID={self.node.node_id}] Exploitation with epsilon={EPSILON:.4f}')
            next_node = self.choose_best_action()

        # Update epsilon after each decision
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        return next_node

    def choose_best_action(self) -> int:
        """
        Encuentra la mejor acción según los valores Q actuales en la Q-table del nodo.
        """
        self.initialize_or_update_q_table()

        neighbors_q_values = self.q_table[self.node.node_id]
        return min(neighbors_q_values, key=neighbors_q_values.get)

    def initialize_or_update_q_table(self) -> None:
        """
        Asegura que la Q-table del nodo esté inicializada o actualizada como una matriz,
        donde cada vecino tiene un valor Q asociado. Si un vecino no tiene un valor Q,
        se inicializa con un valor por defecto.
        """
        if self.node.node_id not in self.q_table:
            self.q_table[self.node.node_id] = {}

        for neighbor in self.node.network.get_neighbors(self.node.node_id):
            if neighbor not in self.q_table[self.node.node_id]:
                self.q_table[self.node.node_id][neighbor] = 0.0

    def estimate_remaining_time(self, next_node) -> float:
        """
        Estima el tiempo restante a partir del valor Q del nodo siguiente.
        Si no hay valores Q asociados, retorna infinito.
        """
        if self.node.node_id not in self.q_table or next_node not in self.q_table[self.node.node_id]:
            return float('inf')

        return self.q_table[self.node.node_id][next_node]

    def update_q_table_with_incomplete_info(self, next_node, estimated_time_remaining) -> None:
        """Actualiza la Q-table para el estado-acción actual usando información incompleta."""

        self.initialize_or_update_q_table()

        current_q = self.q_table[self.node.node_id].get(next_node, 0.0)  # Valor Q actual para el nodo
        updated_q = current_q + ALPHA * (estimated_time_remaining - current_q)

        self.q_table[self.node.node_id][next_node] = updated_q

    def send_packet(self, to_node_id, packet) -> None:
        """
        Envia un paquete al nodo destino utilizando la red.
        """
        packet.hops += 1
        packet.time += 1
        packet.from_node_id = self.node.node_id

        print(f'[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n')
        self.node.network.send(self.node.node_id, to_node_id, packet)

    def __str__(self) -> str:
        return f"Node(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()

class SenderQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)

    def start_episode(self, episode_number) -> None:
        """Initiates an episode by creating a packet and sending it to chosen node."""

        packet = Packet(
            episode_number=episode_number,
            type=PacketType.PACKET_HOP,
            from_node_id=self.node.node_id
        )

        self.initialize_or_update_q_table()

        next_node = self.select_next_node()

        estimated_time_remaining = self.estimate_remaining_time(next_node)

        callback_chain_step = CallbackChainStep(previous_hop_node=None,
                                                next_hop_node=next_node,
                                                send_timestamp=get_current_time(),
                                                estimated_time=estimated_time_remaining)

        self.callback_stack.append(callback_chain_step)

        self.send_packet(next_node, packet)

    def handle_packet_hop(self, packet) -> None:
        next_node = self.select_next_node()

        estimated_time_remaining = self.estimate_remaining_time(next_node)

        self.update_q_table_with_incomplete_info(
            next_node=next_node,
            estimated_time_remaining=estimated_time_remaining
        )

        callback_chain_step = CallbackChainStep(previous_hop_node=packet.from_node_id,
                                                next_hop_node=next_node,
                                                send_timestamp=get_current_time(),
                                                estimated_time=estimated_time_remaining)

        self.callback_stack.append(callback_chain_step)

        print(f'[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}')

        self.send_packet(next_node, packet)

    def handle_echo_callback(self, packet) -> None:
        callback_data = self.callback_stack.pop()

        self.update_q_value(
            next_node=callback_data.next_hop_node,
            s=get_current_time() - callback_data.send_timestamp,
            t=callback_data.estimated_time,
        )

        if self.callback_stack:
            self.send_packet(callback_data.previous_hop_node, packet)
            return

        print(f'\n[Node_ID={self.node.node_id}] Episode {packet.episode_number} finished. Q-Table: {self.q_table}')

    def __str__(self) -> str:
        return f"SenderNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

class IntermediateQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)

    def start_episode(self, episode_number):
        raise NotImplementedError("Intermediate node is not supposed to start an episode")

    def handle_packet_hop(self, packet):
        self.initialize_or_update_q_table()

        if self.assigned_function is None:
            self.assign_function(packet)

        if self.assigned_function == packet.next_function():
            print(f'[Node_ID={self.node.node_id}] Removing function {self.assigned_function} from functions to process')
            packet.remove_next_function()

        if packet.is_sequence_completed():
            print(f'[Node_ID={self.node.node_id}] Function sequence is complete! Initiating full echo callback')
            self.initiate_full_echo_callback(packet)
            return

        next_node = self.select_next_node()
        print(f'[Node_ID={self.node.node_id}] Next node is {next_node}')

        if next_node is not None:
            estimated_time_remaining = self.estimate_remaining_time(next_node)

            self.update_q_table_with_incomplete_info(
                next_node=next_node,
                estimated_time_remaining=estimated_time_remaining
            )

            callback_chain_step = CallbackChainStep(previous_hop_node=packet.from_node_id,
                                                    next_hop_node=next_node,
                                                    send_timestamp=get_current_time(),
                                                    estimated_time=estimated_time_remaining)

            self.callback_stack.append(callback_chain_step)

            print(f'[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}')

            self.send_packet(next_node, packet)

    def handle_echo_callback(self, packet):
        """Maneja el callback cuando regresa el paquete."""
        callback_data = self.callback_stack.pop()

        self.update_q_value(
            next_node=callback_data.next_hop_node,
            s=get_current_time() - callback_data.send_timestamp,
            t=callback_data.estimated_time,
        )

        self.send_packet(callback_data.previous_hop_node, packet)

    def initiate_full_echo_callback(self, packet):
        """Inicia el proceso de full echo callback hacia el nodo anterior."""

        callback_packet = Packet(
            episode_number=packet.episode_number,
            from_node_id=self.node.node_id,
            type=PacketType.CALLBACK
        )

        self.send_packet(packet.from_node_id, callback_packet)

    def assign_function(self, packet):
        """Asigna la función menos utilizada basada en los contadores del paquete."""
        min_assignments = min(packet.function_counters.values())

        least_assigned_functions = [
            func for func, count in packet.function_counters.items() if count == min_assignments
        ]

        if len(least_assigned_functions) == 1:
            function_to_assign = least_assigned_functions[0]
        else:
            function_to_assign = random.choice(least_assigned_functions)

        print(f'[Node_ID={self.node.node_id}] Node has no function, assigning function {function_to_assign}')
        self.assigned_function = function_to_assign
        packet.increment_function_counter(function_to_assign)

    def __str__(self):
        return f"IntermediateNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

class PacketType(Enum):
    PACKET_HOP = "PACKET_HOP"
    CALLBACK = "CALLBACK"

class Packet:
    def __init__(self, episode_number, from_node_id, type):
        self.type = type
        self.episode_number = episode_number  # Número de episodio al que pertenece este paquete
        self.from_node_id = from_node_id  # Nodo anterior por el que pasó el paquete
        self.functions_sequence = FUNCTION_SEQ.copy()  # Secuencia de funciones a procesar
        self.function_counters = {func: 0 for func in FUNCTION_SEQ}  # Contadores de funciones asignadas
        self.hops = 0  # Contador de saltos
        self.time = 0  # Tiempo total acumulado del paquete

    def increment_function_counter(self, function):
        """
        Incrementa el contador de asignaciones para una función específica.
        """
        if function not in self.function_counters:
            raise ValueError(f"La función {function} no es válida.")
        self.function_counters[function] += 1

    def is_sequence_completed(self):
        """Revisa si la secuencia de funciones ha sido completamente procesada."""
        return len(self.functions_sequence) == 0

    def next_function(self):
        """Obtiene la próxima función en la secuencia, si existe."""
        return self.functions_sequence[0] if self.functions_sequence else None

    def remove_next_function(self):
        """Elimina la función actual de la secuencia, marcándola como procesada."""
        if self.functions_sequence:
            self.functions_sequence.pop(0)

    def __str__(self):
        functions_sequence_str = [func.value for func in self.functions_sequence]
        function_counters_str = {func.value: count for func, count in self.function_counters.items()}
        return (
            f"Packet("
            f"type={self.type.value}, "
            f"episode_number={self.episode_number}, "
            f"from_node_id={self.from_node_id}, "
            f"functions_sequence={functions_sequence_str}, "
            f"function_counters={function_counters_str}, "
            f"hops={self.hops}, "
            f"time={self.time})"
        )

def get_current_time():
    return time.time()
