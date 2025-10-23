// SPX Dashboard JavaScript
// This file is automatically created by spx_dashboard.py

const { useState, useEffect } = React;

// Simple SVG icons
const Icons = {
    TrendingUp: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('polyline', {points: '23 6 13.5 15.5 8.5 10.5 1 18'}),
        React.createElement('polyline', {points: '17 6 23 6 23 12'})
    ),
    TrendingDown: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('polyline', {points: '23 18 13.5 8.5 8.5 13.5 1 6'}),
        React.createElement('polyline', {points: '17 18 23 18 23 12'})
    ),
    Minus: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('line', {x1: 5, y1: 12, x2: 19, y2: 12})
    ),
    Activity: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('polyline', {points: '22 12 18 12 15 21 9 3 6 12 2 12'})
    ),
    Target: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('circle', {cx: 12, cy: 12, r: 10}),
        React.createElement('circle', {cx: 12, cy: 12, r: 6}),
        React.createElement('circle', {cx: 12, cy: 12, r: 2})
    ),
    CheckCircle: () => React.createElement('svg', {width: 24, height: 24, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('path', {d: 'M22 11.08V12a10 10 0 1 1-5.93-9.14'}),
        React.createElement('polyline', {points: '22 4 12 14.01 9 11.01'})
    ),
    Clock: () => React.createElement('svg', {width: 20, height: 20, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('circle', {cx: 12, cy: 12, r: 10}),
        React.createElement('polyline', {points: '12 6 12 12 16 14'})
    ),
    DollarSign: () => React.createElement('svg', {width: 24, height: 24, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('line', {x1: 12, y1: 1, x2: 12, y2: 23}),
        React.createElement('path', {d: 'M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'})
    ),
    AlertCircle: () => React.createElement('svg', {width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2},
        React.createElement('circle', {cx: 12, cy: 12, r: 10}),
        React.createElement('line', {x1: 12, y1: 8, x2: 12, y2: 12}),
        React.createElement('line', {x1: 12, y1: 16, x2: 12.01, y2: 16})
    )
};

const Dashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeHorizon, setTimeHorizon] = useState(null);
    
    useEffect(() => {
        fetch('dashboard_data.json')
            .then(res => {
                if (!res.ok) throw new Error('Failed to load data');
                return res.json();
            })
            .then(data => {
                console.log('Data loaded:', data);
                setData(data);
                if (data.available_horizons && data.available_horizons.length > 0) {
                    setTimeHorizon(data.available_horizons[data.available_horizons.length - 1]);
                }
                setLoading(false);
            })
            .catch(err => {
                console.error('Error:', err);
                setError(err.message);
                setLoading(false);
            });
    }, []);
    
    if (loading) {
        return React.createElement('div', {className: 'min-h-screen bg-slate-900 flex items-center justify-center'},
            React.createElement('div', {className: 'text-white text-2xl'}, 'Loading dashboard...')
        );
    }
    
    if (error || !data || !timeHorizon) {
        return React.createElement('div', {className: 'min-h-screen bg-slate-900 flex items-center justify-center'},
            React.createElement('div', {className: 'text-red-400 text-xl p-8 text-center'},
                React.createElement('div', {className: 'mb-4'}, '❌ Error loading dashboard'),
                React.createElement('div', {className: 'text-sm text-slate-400'}, error || 'Data file not found'),
                React.createElement('div', {className: 'text-sm text-slate-500 mt-4'}, 'Make sure dashboard_data.json exists')
            )
        );
    }
    
    // Calculate price targets dynamically
    const priceTargets = {};
    data.available_ranges.forEach(rangeKey => {
        const pct = parseInt(rangeKey.replace('pct', '')) / 100;
        priceTargets[rangeKey] = {
            upper: (data.spx_price * (1 + pct)).toFixed(2),
            lower: (data.spx_price * (1 - pct)).toFixed(2)
        };
    });
    
    const getDirectionalSignal = (prob) => {
        if (prob >= 0.70) return {icon: Icons.TrendingUp, color: 'text-green-500', bg: 'bg-green-50', label: 'BULLISH'};
        if (prob >= 0.55) return {icon: Icons.TrendingUp, color: 'text-green-400', bg: 'bg-green-50', label: 'LEAN BULL'};
        if (prob >= 0.45) return {icon: Icons.Minus, color: 'text-gray-500', bg: 'bg-gray-50', label: 'NEUTRAL'};
        if (prob >= 0.30) return {icon: Icons.TrendingDown, color: 'text-red-400', bg: 'bg-red-50', label: 'LEAN BEAR'};
        return {icon: Icons.TrendingDown, color: 'text-red-500', bg: 'bg-red-50', label: 'BEARISH'};
    };
    
    const getConfidenceColor = (prob) => {
        if (prob >= 0.80) return 'text-green-600 font-bold';
        if (prob >= 0.65) return 'text-green-500 font-semibold';
        if (prob >= 0.55) return 'text-yellow-600';
        return 'text-gray-600';
    };
    
    const getRangeColor = (prob) => {
        if (prob >= 0.90) return 'bg-green-500';
        if (prob >= 0.75) return 'bg-yellow-500';
        return 'bg-red-500';
    };
    
    return React.createElement('div', {className: 'min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4 sm:p-8'},
        React.createElement('div', {className: 'max-w-7xl mx-auto space-y-6'},
            
            // Header
            React.createElement('div', {className: 'flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4'},
                React.createElement('div', {},
                    React.createElement('h1', {className: 'text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent'},
                        'SPX Prediction System'
                    ),
                    React.createElement('p', {className: 'text-slate-400 mt-1 text-sm sm:text-base'}, 
                        'Machine Learning • Fibonacci Horizons (8, 13, 21, 34d)'
                    )
                ),
                React.createElement('div', {className: 'text-right'},
                    React.createElement('div', {className: 'text-sm text-slate-400'}, 
                        data.current_date + ' ' + (data.current_time || new Date().toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit', timeZone: 'America/New_York'}) + ' ET')
                    ),
                    React.createElement('div', {className: 'text-2xl sm:text-3xl font-bold text-blue-400'}, 
                        '$' + data.spx_price.toLocaleString()
                    ),
                    React.createElement('div', {className: 'text-xs text-slate-500 mt-1'}, 
                        'Model: $' + data.spx_price_model.toLocaleString() + ' • VIX: ' + data.vix.toFixed(1)
                    )
                )
            ),
            
            // Model Health - DYNAMIC status-based styling
            React.createElement('div', {
                className: 'border rounded-xl p-4 ' +
                    (data.model_health.status === 'STRONG' ? 'bg-gradient-to-r from-green-500/10 to-blue-500/10 border-green-500/20' :
                     data.model_health.status === 'GOOD' ? 'bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-blue-500/20' :
                     data.model_health.status === 'FAIR' ? 'bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border-yellow-500/20' :
                     'bg-gradient-to-r from-red-500/10 to-orange-500/10 border-red-500/20')
            },
                React.createElement('div', {className: 'flex items-center justify-between flex-wrap gap-2'},
                    React.createElement('div', {className: 'flex items-center gap-3'},
                        React.createElement(Icons.CheckCircle),
                        React.createElement('div', {},
                            React.createElement('div', {
                                className: 'font-semibold ' +
                                    (data.model_health.status === 'STRONG' ? 'text-green-400' :
                                     data.model_health.status === 'GOOD' ? 'text-blue-400' :
                                     data.model_health.status === 'FAIR' ? 'text-yellow-400' :
                                     'text-red-400')
                            }, 
                                'Model Status: ' + data.model_health.status
                            ),
                            React.createElement('div', {className: 'text-xs sm:text-sm text-slate-400'},
                                data.model_health.message || 
                                (data.model_health.test_accuracy ? 
                                    'Accuracy: ' + (data.model_health.test_accuracy * 100).toFixed(1) + '% ± ' +
                                    (data.model_health.std_dev * 100).toFixed(1) + '% • Gap: ' +
                                    (data.model_health.gap >= 0 ? '+' : '') + (data.model_health.gap * 100).toFixed(1) + '%' 
                                : 'No metrics available')
                            )
                        )
                    ),
                    React.createElement('div', {className: 'text-xs text-slate-400'},
                        'Updated: ' + data.current_date
                    )
                )
            ),
            
            // Time Horizon Selector - DYNAMIC with DTE
            React.createElement('div', {className: 'flex gap-2 bg-slate-800/50 p-2 rounded-xl border border-slate-700 w-full overflow-x-auto'},
                data.available_horizons.map(horizon =>
                    React.createElement('button', {
                        key: horizon,
                        onClick: () => setTimeHorizon(horizon),
                        className: 'px-4 sm:px-6 py-2 rounded-lg font-semibold transition-all whitespace-nowrap flex-shrink-0 ' +
                            (timeHorizon === horizon 
                                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                                : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700')
                    }, 
                        React.createElement('div', {className: 'flex flex-col items-center'},
                            React.createElement('div', {}, horizon.toUpperCase()),
                            data.dte_mapping && React.createElement('div', {className: 'text-xs opacity-70'}, 
                                '≈' + data.dte_mapping[horizon] + ' DTE'
                            )
                        )
                    )
                )
            ),
            
            // Main content grid
            React.createElement('div', {className: 'grid grid-cols-1 lg:grid-cols-2 gap-6'},
                
                // Directional panel
                React.createElement('div', {className: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 space-y-4'},
                    React.createElement('div', {className: 'flex items-center justify-between'},
                        React.createElement('h2', {className: 'text-xl font-bold flex items-center gap-2'},
                            React.createElement(Icons.Activity),
                            'Directional Signal'
                        ),
                        React.createElement(Icons.Clock)
                    ),
                    Object.entries(data.directional).map(([period, pred]) => {
                        const signal = getDirectionalSignal(pred.prob);
                        return React.createElement('div', {
                            key: period,
                            className: 'p-4 rounded-lg border transition-all ' + 
                                (period === timeHorizon 
                                    ? 'border-blue-500 bg-blue-500/10 scale-105'
                                    : 'border-slate-700 bg-slate-800/30')
                        },
                            React.createElement('div', {className: 'flex items-center justify-between mb-2'},
                                React.createElement('div', {className: 'flex items-center gap-3'},
                                    React.createElement('div', {className: 'p-2 rounded-lg ' + signal.bg},
                                        React.createElement(signal.icon)
                                    ),
                                    React.createElement('div', {},
                                        React.createElement('div', {className: 'font-semibold'}, period.toUpperCase()),
                                        React.createElement('div', {className: 'text-xs ' + signal.color}, signal.label),
                                        data.dte_mapping && React.createElement('div', {className: 'text-xs text-slate-500'}, 
                                            '≈' + data.dte_mapping[period] + ' DTE'
                                        )
                                    )
                                ),
                                React.createElement('div', {className: 'text-2xl font-bold ' + getConfidenceColor(pred.prob)},
                                    (pred.prob * 100).toFixed(1) + '%'
                                )
                            ),
                            React.createElement('div', {className: 'relative h-2 bg-slate-700 rounded-full overflow-hidden'},
                                React.createElement('div', {
                                    className: 'absolute left-0 top-0 h-full transition-all ' +
                                        (pred.prob >= 0.65 ? 'bg-green-500' : 
                                         pred.prob >= 0.55 ? 'bg-yellow-500' : 'bg-gray-500'),
                                    style: {width: (pred.prob * 100) + '%'}
                                }),
                                React.createElement('div', {className: 'absolute left-1/2 top-0 w-0.5 h-full bg-white/30'})
                            ),
                            React.createElement('div', {className: 'flex justify-between text-xs text-slate-400 mt-1'},
                                React.createElement('span', {}, 'Bearish'),
                                React.createElement('span', {className: 'text-white/50'}, '50%'),
                                React.createElement('span', {}, 'Bullish')
                            )
                        );
                    }),
                    React.createElement('div', {className: 'pt-2 border-t border-slate-700 text-sm text-slate-400'},
                        React.createElement('div', {className: 'flex items-start gap-2'},
                            React.createElement(Icons.AlertCircle),
                            React.createElement('span', {}, 'Trade at 65%+ confidence. 21d model is most reliable.')
                        )
                    )
                ),
                
                // Range panel - DYNAMIC
                React.createElement('div', {className: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 space-y-4 max-h-[800px] overflow-y-auto'},
                    React.createElement('div', {className: 'flex items-center justify-between sticky top-0 bg-slate-800/95 pb-2 z-10'},
                        React.createElement('h2', {className: 'text-xl font-bold flex items-center gap-2'},
                            React.createElement(Icons.Target),
                            'Range Probability'
                        ),
                        React.createElement('div', {className: 'text-sm text-slate-400'}, timeHorizon.toUpperCase())
                    ),
                    data.available_ranges.map(threshold => {
                        const pct = threshold.replace('pct', '');
                        const prob = data.range_bound[timeHorizon][threshold];
                        const targets = priceTargets[threshold];
                        
                        return React.createElement('div', {key: threshold, className: 'space-y-2'},
                            React.createElement('div', {className: 'flex items-center justify-between'},
                                React.createElement('div', {},
                                    React.createElement('div', {className: 'font-semibold text-lg'}, '±' + pct + '% Range'),
                                    React.createElement('div', {className: 'text-xs text-slate-400'}, 
                                        '$' + targets.lower + ' - $' + targets.upper
                                    )
                                ),
                                React.createElement('div', {className: 'text-2xl font-bold ' + getConfidenceColor(prob)},
                                    (prob * 100).toFixed(1) + '%'
                                )
                            ),
                            React.createElement('div', {className: 'h-3 bg-slate-700 rounded-full overflow-hidden'},
                                React.createElement('div', {
                                    className: 'h-full transition-all ' + getRangeColor(prob),
                                    style: {width: (prob * 100) + '%'}
                                })
                            )
                        );
                    }),
                    React.createElement('div', {className: 'pt-2 border-t border-slate-700 text-sm text-slate-400'},
                        React.createElement('div', {className: 'flex items-start gap-2'},
                            React.createElement(Icons.AlertCircle),
                            React.createElement('span', {}, 'Wider ranges = higher reliability. 21d ±5-13% is optimal.')
                        )
                    )
                )
            ),
            
            // Trade Signals and Feature Importance - Side by Side
            React.createElement('div', {className: 'grid grid-cols-1 lg:grid-cols-2 gap-6'},
                // Trade Signals
                React.createElement('div', {className: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6'},
                    React.createElement('h2', {className: 'text-xl font-bold mb-4 flex items-center gap-2'},
                        React.createElement(Icons.DollarSign),
                        'Recommended Trades'
                    ),
                    React.createElement('div', {className: 'space-y-3'},
                        data.trade_signals.map((trade, idx) =>
                            React.createElement('div', {
                                key: idx,
                                className: 'p-4 rounded-lg border transition-all ' +
                                    (trade.action === 'SELL' 
                                        ? 'border-green-500/30 bg-green-500/5 hover:bg-green-500/10'
                                        : 'border-yellow-500/30 bg-yellow-500/5')
                            },
                                React.createElement('div', {className: 'flex items-start justify-between flex-wrap gap-4'},
                                    React.createElement('div', {className: 'flex-1 min-w-0'},
                                        React.createElement('div', {className: 'flex items-center gap-3 mb-2 flex-wrap'},
                                            React.createElement('div', {
                                                className: 'px-3 py-1 rounded-full text-sm font-bold ' +
                                                    (trade.action === 'SELL' 
                                                        ? 'bg-green-500/20 text-green-400'
                                                        : 'bg-yellow-500/20 text-yellow-400')
                                            }, trade.action),
                                            React.createElement('div', {className: 'font-bold text-lg'}, trade.type),
                                            React.createElement('div', {className: 'text-slate-400 text-sm'}, 
                                                '≈' + (data.dte_mapping[trade.dte + 'd'] || trade.dte) + ' DTE'
                                            )
                                        ),
                                        React.createElement('div', {className: 'grid grid-cols-2 gap-x-6 gap-y-1 text-sm mb-2'},
                                            React.createElement('div', {className: 'flex justify-between'},
                                                React.createElement('span', {className: 'text-slate-400'}, 'Strikes:'),
                                                React.createElement('span', {className: 'font-mono text-white'}, trade.strikes)
                                            ),
                                            React.createElement('div', {className: 'flex justify-between'},
                                                React.createElement('span', {className: 'text-slate-400'}, 'Credit:'),
                                                React.createElement('span', {className: 'font-mono text-green-400'}, trade.credit)
                                            ),
                                            React.createElement('div', {className: 'flex justify-between'},
                                                React.createElement('span', {className: 'text-slate-400'}, 'Max Risk:'),
                                                React.createElement('span', {className: 'font-mono text-red-400'}, trade.risk)
                                            ),
                                            React.createElement('div', {className: 'flex justify-between'},
                                                React.createElement('span', {className: 'text-slate-400'}, 'ROI:'),
                                                React.createElement('span', {className: 'font-mono text-blue-400'}, trade.roi)
                                            )
                                        ),
                                        React.createElement('div', {className: 'text-sm text-slate-300 italic border-l-2 border-slate-600 pl-3'},
                                            trade.rationale
                                        )
                                    ),
                                    React.createElement('div', {className: 'text-right'},
                                        React.createElement('div', {className: 'text-xs text-slate-400 mb-1'}, 'Confidence'),
                                        React.createElement('div', {className: 'text-3xl font-bold ' + getConfidenceColor(trade.confidence)},
                                            (trade.confidence * 100).toFixed(0) + '%'
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                
                // Feature Importance
                React.createElement('div', {className: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6'},
                    React.createElement('h2', {className: 'text-xl font-bold mb-4 flex items-center gap-2'},
                        React.createElement(Icons.Activity),
                        'Key Signals (Deployed Feature Importances)'
                    ),
                    React.createElement('div', {className: 'space-y-3'},
                        Object.entries(data.top_features).map(([feature, importance]) =>
                            React.createElement('div', {key: feature, className: 'space-y-1'},
                                React.createElement('div', {className: 'flex items-center justify-between text-sm'},
                                    React.createElement('span', {className: 'text-slate-300 font-mono'}, feature),
                                    React.createElement('span', {className: 'text-blue-400 font-bold'}, 
                                        (importance * 100).toFixed(1) + '%'
                                    )
                                ),
                                React.createElement('div', {className: 'h-2 bg-slate-700 rounded-full overflow-hidden'},
                                    React.createElement('div', {
                                        className: 'h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all',
                                        style: {width: (importance * 100) + '%'}
                                    })
                                )
                            )
                        )
                    ),
                    React.createElement('div', {className: 'mt-4 pt-4 border-t border-slate-700 text-xs text-slate-400'},
                        'Top predictive features from current model training'
                    )
                )
            ),
            
            // Footer
            React.createElement('div', {className: 'text-center text-slate-500 text-sm py-4'},
                React.createElement('p', {}, 'SPX Predictor • Fibonacci Horizons (8, 13, 21, 34 trading days)'),
                React.createElement('p', {className: 'mt-1 text-xs'}, 'Model performance varies by market regime • Use proper risk management')
            )
        )
    );
};

// Render
ReactDOM.render(React.createElement(Dashboard), document.getElementById('root'));