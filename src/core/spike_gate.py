import logging,pandas as pd
logger=logging.getLogger(__name__)

class SpikeGate:
    def __init__(self,mode='safe'):
        from config import SPIKE_GATE_CONFIG
        self.mode=mode;self.config=SPIKE_GATE_CONFIG['modes'][self.mode]
        self.override_actions=SPIKE_GATE_CONFIG['override_actions'];self.override_log=[]
        logger.info(f"SpikeGate: {self.mode.upper()} mode")
        if not self.config.get('override_enabled',False):logger.info("  Override DISABLED (log only)")
    
    def check_and_override(self,forecast,anomaly_score,current_vix,regime):
        result=forecast.copy()
        result['spike_gate_triggered']=False;result['spike_gate_action']=None
        result['spike_gate_original_direction']=None;result['spike_gate_original_magnitude']=None;result['spike_gate_confidence_boost']=0.0
        if anomaly_score is None:return result
        if self.config.get('log_only',False):
            self._log_potential_override(forecast,anomaly_score,regime);return result
        if self.config.get('regime_filter')is not None:
            if regime not in self.config['regime_filter']:return result
        min_score=self.config.get('min_anomaly_score',0.85)
        if anomaly_score<min_score:return result
        action=self._determine_action(forecast,anomaly_score)
        if action is None:return result
        result=self._execute_override(forecast,action,anomaly_score,current_vix)
        self._log_override(result,forecast,anomaly_score,regime)
        return result
    
    def _determine_action(self,forecast,anomaly_score):
        direction=forecast['direction']
        for action_name,action_config in self.override_actions.items():
            min_score=action_config.get('min_score',0.85)
            if anomaly_score<min_score:continue
            if 'direction_conflict'in action_config:
                if direction==action_config['direction_conflict']:return action_name
            if 'direction_match'in action_config:
                if direction==action_config['direction_match']:return action_name
        return None
    
    def _execute_override(self,original_forecast,action,anomaly_score,current_vix):
        result=original_forecast.copy();action_config=self.override_actions[action]
        result['spike_gate_triggered']=True;result['spike_gate_action']=action
        result['spike_gate_original_direction']=original_forecast['direction']
        result['spike_gate_original_magnitude']=original_forecast['magnitude_pct']
        if action=='OVERRIDE_TO_UP':
            orig_mag=abs(original_forecast['magnitude_pct']);new_mag=orig_mag*action_config['magnitude_multiplier']
            result['direction']='UP';result['magnitude_pct']=new_mag;result['expected_vix']=current_vix*(1+new_mag/100)
            result['direction_confidence']=action_config['confidence']
            result['spike_gate_confidence_boost']=action_config['confidence']-original_forecast['direction_confidence']
            logger.warning(f"🔴 SPIKE GATE: DOWN→UP (score={anomaly_score:.3f})")
        elif action=='OVERRIDE_TO_NEUTRAL':
            result['direction']='NEUTRAL';result['magnitude_pct']=0.0;result['expected_vix']=current_vix
            result['direction_confidence']=action_config['confidence']
            result['spike_gate_confidence_boost']=action_config['confidence']-original_forecast['direction_confidence']
            logger.warning(f"🟠 SPIKE GATE: DOWN→NEUTRAL (score={anomaly_score:.3f})")
        elif action=='BOOST_CONFIRM':
            orig_mag=original_forecast['magnitude_pct'];new_mag=orig_mag*action_config['magnitude_multiplier']
            result['magnitude_pct']=new_mag;result['expected_vix']=current_vix*(1+new_mag/100)
            orig_conf=original_forecast['direction_confidence']
            new_conf=min(orig_conf*action_config['confidence_multiplier'],0.95)
            result['direction_confidence']=new_conf;result['spike_gate_confidence_boost']=new_conf-orig_conf
            logger.info(f"🟢 SPIKE GATE: UP boost (score={anomaly_score:.3f})")
        return result
    
    def _log_potential_override(self,forecast,anomaly_score,regime):
        action=self._determine_action(forecast,anomaly_score)
        if action is not None:
            log_entry={'timestamp':pd.Timestamp.now(),'anomaly_score':anomaly_score,'regime':regime,'original_direction':forecast['direction'],'original_magnitude':forecast['magnitude_pct'],'would_trigger_action':action,'executed':False}
            self.override_log.append(log_entry)
            logger.info(f"💡 [SAFE MODE] Would trigger {action} (score={anomaly_score:.3f}, dir={forecast['direction']}, regime={regime})")
    
    def _log_override(self,result,original,anomaly_score,regime):
        log_entry={'timestamp':pd.Timestamp.now(),'anomaly_score':anomaly_score,'regime':regime,'original_direction':original['direction'],'original_magnitude':original['magnitude_pct'],'new_direction':result['direction'],'new_magnitude':result['magnitude_pct'],'action':result['spike_gate_action'],'executed':True}
        self.override_log.append(log_entry)
    
    def get_override_stats(self):
        if not self.override_log:return {'total_checks':0,'overrides_executed':0,'overrides_potential':0}
        executed=sum(1 for log in self.override_log if log.get('executed',False))
        potential=sum(1 for log in self.override_log if not log.get('executed',False))
        action_counts={}
        for log in self.override_log:
            if log.get('executed',False):
                action=log.get('action','UNKNOWN');action_counts[action]=action_counts.get(action,0)+1
        return {'total_checks':len(self.override_log),'overrides_executed':executed,'overrides_potential':potential,'by_action':action_counts,'recent_log':self.override_log[-10:]}
