from ..Core.Interfaces import TransitionInfoProvider

import numpy as np


class SEEDEnergyProvider(TransitionInfoProvider):

    name = "SEED Energy Provider"

    def __init__(self):
        super().__init__()

        # Ethernet Constants Metrics (1GB Rate - Active/Idle Toggling with 0BASE-x for Energy Efficient Ethernet, November 2007 IEEE 802.3az Task Force)
        self.ethernet_active_w = 1.217
        self.ethernet_idle_w = 1.010

        # LTE Constants Metrics
        self.lte_p_conn_w = 1.53
        self.lte_p_rb_w = 0.42

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

        signal_strength_dbm = info_t.get("signal_strength_dbm", [])

        used_energy_encstore_j = segment.get("used_energy_encstore", 0.0)
        used_energy_decoding_j = segment.get("used_energy_decoding", 0.0)
        used_energy_display_j = segment.get("used_energy_display", 0.0)

        used_energy_ret = 0.0

        match config.nic:

            case "Eth":
                used_energy_ret = self._energy_estimate_eth(
                    download_period=download_time_s,
                )

            case "LTE" | "5G":

                avg_signal_strength_dbm = (
                    np.mean(signal_strength_dbm)
                    if len(signal_strength_dbm) > 0
                    else -60.0
                )
                avg_throughput_kbps = (
                    np.mean(throughputs_kbps) if len(throughputs_kbps) > 0 else 0.0
                )
                avg_throughput_mbps = avg_throughput_kbps / 1000.0

                used_energy_ret_j = self._energy_estimate_lte(
                    signal_strength_dbm=avg_signal_strength_dbm,
                    downlink_throughput=avg_throughput_mbps,
                    download_period=download_time_s,
                )

        return {
            "used_energy_ret": used_energy_ret_j,
            "used_energy_encstore": used_energy_encstore_j,
            "used_energy_decoding": used_energy_decoding_j,
            "used_energy_display": used_energy_display_j,
        }

    def _energy_estimate_eth(
        self,
        download_period: float,
    ) -> float:

        pass

    def _energy_estimate_lte(
        self,
        signal_strength_dbm: float,
        downlink_throughput: float,
        download_period: float,
    ) -> float:

        p_rx_rf = (1889.0 - 1.11 * signal_strength_dbm) / 1000
        p_rx_bb = (1923.0 + 2.89 * downlink_throughput) / 1000

        second_estimate = self.lte_p_conn_w + self.lte_p_rb_w + p_rx_rf + p_rx_bb

        return second_estimate * download_period
