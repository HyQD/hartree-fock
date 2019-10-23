from quantum_systems.sampler import SampleCollector, Sampler


class TDHFObservableSampler(Sampler):
    energy_key = "energy"
    dipole_keys = {
        0: "dipole_moment_x",
        1: "dipole_moment_y",
        2: "dipole_moment_z",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.energy = self.np.zeros(self.num_samples, dtype=self.np.complex128)

        self.dim = self.system.dipole_moment.shape[0]
        self.dipole_moment = self.np.zeros(
            (self.dim, self.num_samples), dtype=self.np.complex128
        )

    def sample(self, step):
        self.energy[step] = self.solver.compute_energy()

        rho_qp = self.solver.compute_one_body_density_matrix()

        for i in range(self.dim):
            dipole = self.system.dipole_moment[i]
            self.dipole_moment[i, step] = self.np.trace(rho_qp @ dipole)

    def dump(self, samples):
        samples[self.energy_key] = self.energy

        for i in range(self.dim):
            samples[self.dipole_keys[i]] = self.dipole_moment[i]

        return samples


class TDHFSampleAll(SampleCollector):
    def __init__(self, solver, num_samples, np):
        super().__init__(
            [TDHFObservableSampler(solver, num_samples, np)], np=np
        )
