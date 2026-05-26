from ..Core.Interfaces import TransitionInfoProvider

import numpy as np


class SEEDEnergyInfoProvider(TransitionInfoProvider):

    name = "SEED Energy Provider"

    def __init__(self):
        super().__init__()

        # Ethernet Constants Metrics (1GB Rate - Active/Idle Toggling with 0BASE-x for Energy Efficient Ethernet, November 2007 IEEE 802.3az Task Force)
        self.ethernet_active_w = 1.217
        self.ethernet_idle_w = 1.010

        # LTE Constants Metrics
        self.lte_p_conn_w = 1.53
        self.lte_p_rb_w = 0.42

        # Fallbacks
        self.default_signal_strength_dbm = -60.0
        self.default_throughput_kbps = 1000.0

    def compute(
        self,
        config,
        state_t,
        action_t,
        segment,
        info_t,
        state_t1,
    ) -> dict[str, float]:
        # SEEDEnergy is in Joules

        download_time_s = info_t.get("download_time_s", 0.0)

        throughputs_kbps = info_t.get("throughput_kbps", [])

        used_energy_encstore_j = segment.get("used_energy_encstore", 0.0)
        used_energy_decoding_j = segment.get("used_energy_decoding", 0.0)
        used_energy_display_j = segment.get("used_energy_display", 0.0)

        used_energy_ret_j = 0.0
        idle_energy_j = 0.0

        match config.nic:

            case "Eth":
                idle_energy_j, used_energy_ret_j = self._energy_estimate_eth(
                    download_period=download_time_s,
                    wait_time_s=info_t.get("wait_time_s", 0.0),
                    throughputs_kbps=throughputs_kbps,
                )

            case "LTE" | "5G":

                avg_throughput_kbps = (
                    np.mean(throughputs_kbps)
                    if len(throughputs_kbps) > 0
                    else self.default_throughput_kbps
                )
                avg_throughput_mbps = avg_throughput_kbps / 1000.0

                used_energy_ret_j = self._energy_estimate_lte(
                    downlink_throughputs=throughputs_kbps,
                    download_period=download_time_s,
                )

        return {
            "used_energy_ret": used_energy_ret_j,
            "used_energy_encstore": used_energy_encstore_j,
            "used_energy_decoding": used_energy_decoding_j,
            "used_energy_display": used_energy_display_j,
            "idle_energy": idle_energy_j,
        }

    def _energy_estimate_eth(
        self,
        download_period: float,
        wait_time_s: float,
        throughputs_kbps: float,
    ) -> float:

        idle_power_j = self.ethernet_idle_w * wait_time_s
        active_power_j = self.ethernet_active_w * download_period

        return idle_power_j, active_power_j

    def _energy_estimate_lte(
        self,
        downlink_throughputs: list[float],
        download_period: float,
    ) -> float:

        tp_arr = np.asarray(downlink_throughputs, dtype=float)

        p_rx_rf = (1889.0 - 1.11 * self.default_signal_strength_dbm) / 1000.0
        p_rx_bb = (1923.0 + 2.89 * tp_arr) / 1000.0

        power_w = self.lte_p_conn_w + self.lte_p_rb_w + p_rx_rf + p_rx_bb

        return float(np.mean(power_w) * download_period)
