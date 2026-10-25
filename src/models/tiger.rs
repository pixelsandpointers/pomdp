use crate::distributions::SparseCat;
use crate::model::{FiniteMDP, FinitePOMDP, MDP, POMDP};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TigerState {
    TigerLeft,
    TigerRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TigerAction {
    Listen,
    OpenLeft,
    OpenRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TigerObservation {
    HearLeft,
    HearRight,
}

/// The classic tiger problem (Kaelbling, Littman & Cassandra, 1998).
///
/// A tiger is behind one of two doors. Listening costs a little and reports the tiger's side
/// correctly with probability `p_listen_correctly`. Opening a door ends the round: the agent is
/// rewarded for escaping or penalised for finding the tiger, and the tiger is placed again at
/// random.
#[derive(Clone, Debug)]
pub struct TigerPOMDP {
    pub r_listen: f64,
    pub r_findtiger: f64,
    pub r_escapetiger: f64,
    pub p_listen_correctly: f64,
    pub discount: f64,
}

impl Default for TigerPOMDP {
    fn default() -> Self {
        Self {
            r_listen: -1.0,
            r_findtiger: -100.0,
            r_escapetiger: 10.0,
            p_listen_correctly: 0.85,
            discount: 0.95,
        }
    }
}

use TigerAction::*;
use TigerObservation::*;
use TigerState::*;

impl MDP for TigerPOMDP {
    type State = TigerState;
    type Action = TigerAction;
    type TransitionDist = SparseCat<TigerState>;
    type InitialStateDist = SparseCat<TigerState>;

    fn transition(&self, s: &TigerState, a: &TigerAction) -> SparseCat<TigerState> {
        match a {
            Listen => SparseCat::new(vec![*s], vec![1.0]),
            OpenLeft | OpenRight => SparseCat::new(vec![TigerLeft, TigerRight], vec![0.5, 0.5]),
        }
    }

    fn reward(&self, s: &TigerState, a: &TigerAction, _sp: &TigerState) -> f64 {
        match (a, s) {
            (Listen, _) => self.r_listen,
            (OpenLeft, TigerLeft) | (OpenRight, TigerRight) => self.r_findtiger,
            (OpenLeft, TigerRight) | (OpenRight, TigerLeft) => self.r_escapetiger,
        }
    }

    fn discount(&self) -> f64 {
        self.discount
    }

    fn initial_state(&self) -> SparseCat<TigerState> {
        SparseCat::new(vec![TigerLeft, TigerRight], vec![0.5, 0.5])
    }
}

impl POMDP for TigerPOMDP {
    type Observation = TigerObservation;
    type ObservationDist = SparseCat<TigerObservation>;

    fn observation(&self, a: &TigerAction, sp: &TigerState) -> SparseCat<TigerObservation> {
        let p = match a {
            Listen => self.p_listen_correctly,
            // Opening a door gives no information.
            OpenLeft | OpenRight => 0.5,
        };
        let (correct, wrong) = match sp {
            TigerLeft => (HearLeft, HearRight),
            TigerRight => (HearRight, HearLeft),
        };
        SparseCat::new(vec![correct, wrong], vec![p, 1.0 - p])
    }
}

impl FiniteMDP for TigerPOMDP {
    fn states(&self) -> Vec<TigerState> {
        vec![TigerLeft, TigerRight]
    }

    fn actions(&self) -> Vec<TigerAction> {
        vec![Listen, OpenLeft, OpenRight]
    }

    fn state_index(&self, s: &TigerState) -> usize {
        *s as usize
    }

    fn action_index(&self, a: &TigerAction) -> usize {
        *a as usize
    }
}

impl FinitePOMDP for TigerPOMDP {
    fn observations(&self) -> Vec<TigerObservation> {
        vec![HearLeft, HearRight]
    }

    fn observation_index(&self, o: &TigerObservation) -> usize {
        *o as usize
    }
}
