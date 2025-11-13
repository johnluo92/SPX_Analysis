/**
 * Unified Data Service V2.1 - Single Source of Truth for Dashboard
 * ================================================================
 * Consolidates 6+ fetch calls per page into 2 shared fetches.
 * 
 * BEFORE: Each subchart loads 2-3 files independently
 * AFTER:  One service loads 2 files, shares across all subcharts
 * 
 * Files loaded:
 * - live_state.json    (15 KB, updates every 15min)
 * - historical.json    (300 KB, static)
 * 
 * NEW IN V2.1:
 * - Enhanced error handling and retry logic
 * - Data validation with detailed warnings
 * - Performance monitoring
 * - Cache management
 * - Debug mode
 * - Fallback mechanisms
 * 
 * Usage:
 *   await window.dataService.init();  // Call once on page load
 *   const anomaly = window.dataService.getAnomalyState();
 *   const market = window.dataService.getMarketState();
 */

class DataService {
    constructor() {
        // Core data
        this.live = null;
        this.historical = null;
        
        // State management
        this.loading = false;
        this.initialized = false;
        this.error = null;
        this.lastUpdate = null;
        this.refreshInterval = null;
        
        // Event system
        this.eventHandlers = {
            'data-loaded': [],
            'data-refreshed': [],
            'data-error': [],
            'init-complete': [],
            'refresh-start': [],
            'refresh-complete': []
        };
        
        // Configuration
        this.config = {
            retryAttempts: 3,
            retryDelay: 1000, // ms
            timeout: 30000, // ms
            debugMode: false,
            enableCache: true,
            cacheDuration: 5 * 60 * 1000, // 5 minutes
            baseUrl: '../../json_data/'
        };
        
        // Performance tracking
        this.performance = {
            initTime: null,
            lastRefreshTime: null,
            totalRefreshes: 0,
            failedRefreshes: 0,
            averageLoadTime: null
        };
        
        // Cache management
        this.cache = {
            live: null,
            historical: null,
            liveTimestamp: null,
            historicalTimestamp: null
        };
        
        // Validation rules
        this.validationRules = {
            live: {
                required: ['schema_version', 'market', 'anomaly', 'persistence'],
                types: {
                    'anomaly.ensemble_score': 'number',
                    'market.vix': 'number',
                    'market.spx': 'number'
                }
            },
            historical: {
                required: ['schema_version', 'historical', 'attribution'],
                types: {
                    'historical.dates': 'array',
                    'historical.ensemble_scores': 'array'
                }
            }
        };
    }
    
    /**
     * Initialize data service (call once on page load)
     * @param {Object} options - Configuration options
     * @returns {Promise<void>}
     */
    async init(options = {}) {
        if (this.loading) {
            this._log('Already loading...', 'warn');
            return this._waitForInit();
        }
        
        if (this.initialized) {
            this._log('Already initialized', 'info');
            return;
        }
        
        // Merge options
        Object.assign(this.config, options);
        
        this.loading = true;
        this.error = null;
        
        this._log('Initializing...', 'info');
        
        try {
            const startTime = performance.now();
            
            // Check cache first
            if (this.config.enableCache && this._isCacheValid()) {
                this._log('Using cached data', 'info');
                this.live = this.cache.live;
                this.historical = this.cache.historical;
            } else {
                // Load both files in parallel with retry logic
                const [liveData, historicalData] = await Promise.all([
                    this._fetchWithRetry('live_state.json', true),
                    this._fetchWithRetry('historical.json', false)
                ]);
                
                this.live = liveData;
                this.historical = historicalData;
                
                // Update cache
                if (this.config.enableCache) {
                    this._updateCache();
                }
            }
            
            this.lastUpdate = new Date();
            this.initialized = true;
            
            const loadTime = (performance.now() - startTime).toFixed(0);
            this.performance.initTime = loadTime;
            
            this._log('Initialized successfully', 'success', {
                schema: this.live.schema_version,
                trainingWindow: this.historical.training_window,
                loadTime: `${loadTime}ms`,
                timestamp: this.lastUpdate.toISOString()
            });
            
            // Validate data
            const validationResults = this._validate();
            if (validationResults.errors.length > 0) {
                this._log('Validation errors detected', 'warn', validationResults.errors);
            }
            
            // Emit events
            this._emit('data-loaded', { live: this.live, historical: this.historical });
            this._emit('init-complete', { loadTime, validationResults });
            
        } catch (error) {
            this.error = error;
            this.initialized = false;
            this._log('Initialization failed', 'error', error);
            this._emit('data-error', { error, phase: 'init' });
            throw error;
        } finally {
            this.loading = false;
        }
    }
    
    /**
     * Wait for initialization to complete
     * @private
     */
    async _waitForInit() {
        return new Promise((resolve, reject) => {
            const checkInterval = setInterval(() => {
                if (!this.loading) {
                    clearInterval(checkInterval);
                    if (this.initialized) {
                        resolve();
                    } else {
                        reject(new Error('Initialization failed'));
                    }
                }
            }, 100);
            
            // Timeout after 30 seconds
            setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error('Initialization timeout'));
            }, 30000);
        });
    }
    
    /**
     * Fetch with retry logic
     * @private
     */
    async _fetchWithRetry(filename, bustCache = false) {
        let lastError;
        
        for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
            try {
                const url = this.config.baseUrl + filename + (bustCache ? '?t=' + Date.now() : '');
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
                
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                return data;
                
            } catch (error) {
                lastError = error;
                this._log(`Fetch attempt ${attempt}/${this.config.retryAttempts} failed: ${filename}`, 'warn', error);
                
                if (attempt < this.config.retryAttempts) {
                    await this._sleep(this.config.retryDelay * attempt);
                }
            }
        }
        
        throw new Error(`Failed to fetch ${filename} after ${this.config.retryAttempts} attempts: ${lastError.message}`);
    }
    
    /**
     * Sleep utility
     * @private
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Refresh live data only (called by auto-refresh)
     * Historical data is static, so we don't reload it
     * @returns {Promise<void>}
     */
    async refresh() {
        if (!this.initialized) {
            throw new Error('Service not initialized. Call init() first.');
        }
        
        this._log('Refreshing live state...', 'info');
        this._emit('refresh-start');
        
        try {
            const startTime = performance.now();
            
            const liveData = await this._fetchWithRetry('live_state.json', true);
            
            // Validate before updating
            const oldLive = this.live;
            this.live = liveData;
            
            const validationResults = this._validateLive();
            if (validationResults.errors.length > 0) {
                this._log('Validation errors in refreshed data', 'warn', validationResults.errors);
                // Rollback if critical errors
                if (validationResults.critical) {
                    this.live = oldLive;
                    throw new Error('Critical validation errors in refreshed data');
                }
            }
            
            // Update cache
            if (this.config.enableCache) {
                this.cache.live = this.live;
                this.cache.liveTimestamp = Date.now();
            }
            
            this.lastUpdate = new Date();
            this.performance.totalRefreshes++;
            
            const refreshTime = (performance.now() - startTime).toFixed(0);
            this.performance.lastRefreshTime = refreshTime;
            
            // Update average load time
            if (this.performance.averageLoadTime === null) {
                this.performance.averageLoadTime = parseFloat(refreshTime);
            } else {
                this.performance.averageLoadTime = 
                    (this.performance.averageLoadTime * 0.8) + (parseFloat(refreshTime) * 0.2);
            }
            
            this._log('Live state refreshed', 'success', {
                timestamp: this.live.market.timestamp,
                score: this.live.anomaly.ensemble_score.toFixed(3),
                refreshTime: `${refreshTime}ms`
            });
            
            // Emit refresh event for subcharts to update
            this._emit('data-refreshed', { live: this.live, refreshTime });
            this._emit('refresh-complete', { success: true, refreshTime });
            
        } catch (error) {
            this.performance.failedRefreshes++;
            this._log('Refresh failed', 'error', error);
            this._emit('data-error', { error, phase: 'refresh' });
            this._emit('refresh-complete', { success: false, error });
            throw error;
        }
    }
    
    /**
     * Start auto-refresh
     * @param {number} intervalMs - Refresh interval in milliseconds (default: 15 min)
     */
    startAutoRefresh(intervalMs = 15 * 60 * 1000) {
        if (this.refreshInterval) {
            this._log('Auto-refresh already running', 'warn');
            return;
        }
        
        this._log(`Auto-refresh started (${intervalMs / 60000} min)`, 'info');
        
        this.refreshInterval = setInterval(async () => {
            try {
                await this.refresh();
            } catch (error) {
                this._log('Auto-refresh error', 'error', error);
                // Continue running despite errors
            }
        }, intervalMs);
    }
    
    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
            this._log('Auto-refresh stopped', 'info');
        }
    }
    
    /**
     * Get current anomaly state
     * @returns {Object|null}
     */
    getAnomalyState() {
        if (!this.live) {
            this._log('No live data available', 'warn');
            return null;
        }
        
        return {
            score: this.live.anomaly.ensemble_score,
            classification: this.live.anomaly.classification,
            activeDetectors: this.live.anomaly.active_detectors,
            detectorScores: this.live.anomaly.detector_scores,
            persistence: this.live.persistence,
            diagnostics: this.live.diagnostics
        };
    }
    
    /**
     * Get current market state
     * @returns {Object|null}
     */
    getMarketState() {
        if (!this.live) {
            this._log('No live data available', 'warn');
            return null;
        }
        return this.live.market;
    }
    
    /**
     * Get persistence state from live data
     * @returns {Object|null}
     */
    getPersistenceState() {
        if (!this.live) {
            this._log('No live data available', 'warn');
            return null;
        }
        return this.live.persistence;
    }
    
    /**
     * Get historical data for charting
     * @returns {Object|null}
     */
    getHistoricalData() {
        if (!this.historical) {
            this._log('No historical data available', 'warn');
            return null;
        }
        
        const hist = this.historical.historical;
        return {
            dates: hist.dates,
            scores: hist.ensemble_scores,
            spx: hist.spx_close,
            forwardReturns: hist.spx_forward_10d,
            regimeStats: hist.regime_stats,
            thresholds: hist.thresholds
        };
    }
    
    /**
     * Get thresholds (with confidence intervals if available)
     * @returns {Object|null}
     */
    getThresholds() {
        if (!this.historical) {
            this._log('No historical data available', 'warn');
            return null;
        }
        
        const thresholdsWithCI = this.historical.thresholds_with_ci;
        const baseThresholds = this.historical.historical.thresholds;
        
        return {
            base: baseThresholds,
            withCI: thresholdsWithCI || baseThresholds,
            hasConfidenceIntervals: thresholdsWithCI && 'moderate_ci' in thresholdsWithCI
        };
    }
    
    /**
     * Get feature attribution for a specific detector
     * @param {string} detectorName - Name of detector (e.g., 'vix_mean_reversion')
     * @returns {Array|null}
     */
    getFeatureAttribution(detectorName) {
        if (!this.historical) {
            this._log('No historical data available', 'warn');
            return null;
        }
        
        const attribution = this.historical.attribution;
        if (!attribution[detectorName]) {
            this._log(`No attribution found for detector: ${detectorName}`, 'warn');
            return null;
        }
        
        return attribution[detectorName];
    }
    
    /**
     * Get detector metadata
     * @returns {Object|null}
     */
    getDetectorMetadata() {
        if (!this.historical) {
            this._log('No historical data available', 'warn');
            return null;
        }
        return this.historical.detector_metadata;
    }
    
    /**
     * Get regime statistics
     * @returns {Object|null}
     */
    getRegimeStats() {
        if (!this.historical) {
            this._log('No historical data available', 'warn');
            return null;
        }
        return this.historical.historical.regime_stats;
    }
    
    /**
     * Check if data is loaded
     * @returns {boolean}
     */
    isLoaded() {
        return this.live !== null && this.historical !== null;
    }
    
    /**
     * Get last update timestamp
     * @returns {Date|null}
     */
    getLastUpdate() {
        return this.lastUpdate;
    }
    
    /**
     * Register event listener
     * @param {string} event - Event name
     * @param {Function} handler - Callback function
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        } else {
            this._log(`Unknown event type: ${event}`, 'warn');
        }
    }
    
    /**
     * Unregister event listener
     * @param {string} event - Event name
     * @param {Function} handler - Callback function
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
        }
    }
    
    /**
     * Emit event to registered handlers
     * @private
     */
    _emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    this._log(`Event handler error (${event})`, 'error', error);
                }
            });
        }
        
        // Also dispatch as DOM event for backwards compatibility
        window.dispatchEvent(new CustomEvent(event, { detail: data }));
    }
    
    /**
     * Validate loaded data structure
     * @private
     */
    _validate() {
        const errors = [];
        const warnings = [];
        let critical = false;
        
        // Validate live state
        const liveResult = this._validateLive();
        errors.push(...liveResult.errors);
        warnings.push(...liveResult.warnings);
        critical = critical || liveResult.critical;
        
        // Validate historical
        const historicalResult = this._validateHistorical();
        errors.push(...historicalResult.errors);
        warnings.push(...historicalResult.warnings);
        critical = critical || historicalResult.critical;
        
        if (warnings.length > 0) {
            this._log('Data validation warnings', 'warn', warnings);
        }
        
        return { errors, warnings, critical, valid: errors.length === 0 };
    }
    
    /**
     * Validate live state
     * @private
     */
    _validateLive() {
        const errors = [];
        const warnings = [];
        let critical = false;
        
        if (!this.live) {
            errors.push('Live state is null');
            critical = true;
            return { errors, warnings, critical };
        }
        
        // Check required fields
        const rules = this.validationRules.live;
        for (const field of rules.required) {
            if (!this._hasField(this.live, field)) {
                errors.push(`Missing required field: ${field}`);
                critical = true;
            }
        }
        
        // Check types
        for (const [field, expectedType] of Object.entries(rules.types)) {
            const value = this._getField(this.live, field);
            if (value !== undefined && !this._checkType(value, expectedType)) {
                errors.push(`Invalid type for ${field}: expected ${expectedType}, got ${typeof value}`);
            }
        }
        
        // Check anomaly score range
        if (this.live.anomaly && typeof this.live.anomaly.ensemble_score === 'number') {
            const score = this.live.anomaly.ensemble_score;
            if (score < 0 || score > 1) {
                warnings.push(`Anomaly score out of range [0,1]: ${score}`);
            }
        }
        
        return { errors, warnings, critical };
    }
    
    /**
     * Validate historical data
     * @private
     */
    _validateHistorical() {
        const errors = [];
        const warnings = [];
        let critical = false;
        
        if (!this.historical) {
            errors.push('Historical data is null');
            critical = true;
            return { errors, warnings, critical };
        }
        
        // Check required fields
        const rules = this.validationRules.historical;
        for (const field of rules.required) {
            if (!this._hasField(this.historical, field)) {
                errors.push(`Missing required field: ${field}`);
                critical = true;
            }
        }
        
        // Check types
        for (const [field, expectedType] of Object.entries(rules.types)) {
            const value = this._getField(this.historical, field);
            if (value !== undefined && !this._checkType(value, expectedType)) {
                errors.push(`Invalid type for ${field}: expected ${expectedType}, got ${typeof value}`);
            }
        }
        
        // Check array lengths match
        if (this.historical.historical) {
            const hist = this.historical.historical;
            if (hist.dates && hist.ensemble_scores) {
                if (hist.dates.length !== hist.ensemble_scores.length) {
                    warnings.push(`Array length mismatch: dates(${hist.dates.length}) vs scores(${hist.ensemble_scores.length})`);
                }
            }
        }
        
        return { errors, warnings, critical };
    }
    
    /**
     * Check if object has nested field
     * @private
     */
    _hasField(obj, path) {
        const parts = path.split('.');
        let current = obj;
        for (const part of parts) {
            if (current === null || current === undefined || !(part in current)) {
                return false;
            }
            current = current[part];
        }
        return true;
    }
    
    /**
     * Get nested field from object
     * @private
     */
    _getField(obj, path) {
        const parts = path.split('.');
        let current = obj;
        for (const part of parts) {
            if (current === null || current === undefined) {
                return undefined;
            }
            current = current[part];
        }
        return current;
    }
    
    /**
     * Check value type
     * @private
     */
    _checkType(value, expectedType) {
        if (expectedType === 'array') {
            return Array.isArray(value);
        }
        return typeof value === expectedType;
    }
    
    /**
     * Check if cache is valid
     * @private
     */
    _isCacheValid() {
        if (!this.cache.live || !this.cache.historical) {
            return false;
        }
        
        const now = Date.now();
        const liveAge = now - (this.cache.liveTimestamp || 0);
        const historicalAge = now - (this.cache.historicalTimestamp || 0);
        
        return liveAge < this.config.cacheDuration && 
               historicalAge < this.config.cacheDuration;
    }
    
    /**
     * Update cache
     * @private
     */
    _updateCache() {
        this.cache.live = this.live;
        this.cache.historical = this.historical;
        this.cache.liveTimestamp = Date.now();
        this.cache.historicalTimestamp = Date.now();
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.live = null;
        this.cache.historical = null;
        this.cache.liveTimestamp = null;
        this.cache.historicalTimestamp = null;
        this._log('Cache cleared', 'info');
    }
    
    /**
     * Enable debug mode
     */
    enableDebug() {
        this.config.debugMode = true;
        this._log('Debug mode enabled', 'info');
    }
    
    /**
     * Disable debug mode
     */
    disableDebug() {
        this.config.debugMode = false;
    }
    
    /**
     * Logging utility
     * @private
     */
    _log(message, level = 'info', data = null) {
        const prefix = {
            'info': 'ℹ️',
            'success': '✅',
            'warn': '⚠️',
            'error': '❌'
        }[level] || 'ℹ️';
        
        const logFn = level === 'error' ? console.error : 
                     level === 'warn' ? console.warn : 
                     console.log;
        
        if (this.config.debugMode || level === 'error' || level === 'success') {
            if (data) {
                logFn(`${prefix} DataService: ${message}`, data);
            } else {
                logFn(`${prefix} DataService: ${message}`);
            }
        }
    }
    
    /**
     * Get debug info
     * @returns {Object}
     */
    getDebugInfo() {
        return {
            // State
            initialized: this.initialized,
            loaded: this.isLoaded(),
            loading: this.loading,
            error: this.error,
            lastUpdate: this.lastUpdate,
            autoRefresh: this.refreshInterval !== null,
            
            // Schemas
            liveSchema: this.live?.schema_version,
            historicalSchema: this.historical?.schema_version,
            
            // Performance
            performance: {
                ...this.performance,
                averageLoadTime: this.performance.averageLoadTime?.toFixed(2) + 'ms'
            },
            
            // Cache
            cache: {
                enabled: this.config.enableCache,
                liveAge: this.cache.liveTimestamp ? 
                    ((Date.now() - this.cache.liveTimestamp) / 1000).toFixed(1) + 's' : null,
                historicalAge: this.cache.historicalTimestamp ? 
                    ((Date.now() - this.cache.historicalTimestamp) / 1000).toFixed(1) + 's' : null
            },
            
            // Memory
            memory: this._estimateMemoryUsage(),
            
            // Config
            config: this.config
        };
    }
    
    /**
     * Estimate memory usage (rough approximation)
     * @private
     */
    _estimateMemoryUsage() {
        try {
            const liveSize = JSON.stringify(this.live || {}).length;
            const historicalSize = JSON.stringify(this.historical || {}).length;
            return {
                live: `${(liveSize / 1024).toFixed(1)} KB`,
                historical: `${(historicalSize / 1024).toFixed(1)} KB`,
                total: `${((liveSize + historicalSize) / 1024).toFixed(1)} KB`
            };
        } catch {
            return { error: 'Unable to estimate' };
        }
    }
    
    /**
     * Force reload data (bypass cache)
     */
    async forceReload() {
        this._log('Force reload initiated', 'info');
        this.clearCache();
        this.initialized = false;
        await this.init();
    }
    
    /**
     * Get health status
     * @returns {Object}
     */
    getHealthStatus() {
        return {
            status: this.initialized && this.isLoaded() && !this.error ? 'healthy' : 'unhealthy',
            initialized: this.initialized,
            dataLoaded: this.isLoaded(),
            lastUpdate: this.lastUpdate,
            error: this.error,
            performance: {
                totalRefreshes: this.performance.totalRefreshes,
                failedRefreshes: this.performance.failedRefreshes,
                successRate: this.performance.totalRefreshes > 0 ? 
                    ((this.performance.totalRefreshes - this.performance.failedRefreshes) / this.performance.totalRefreshes * 100).toFixed(1) + '%' : 
                    'N/A'
            }
        };
    }
}

// Create singleton instance
window.dataService = new DataService();

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataService;
}

// Log version
console.log('📦 DataService loaded (v2.1.0)');

// Expose debug globally for console access
window.debugDataService = () => {
    console.log('='.repeat(80));
    console.log('DATA SERVICE DEBUG INFO');
    console.log('='.repeat(80));
    console.log(window.dataService.getDebugInfo());
    console.log('\nHealth Status:');
    console.log(window.dataService.getHealthStatus());
    console.log('='.repeat(80));
};

console.log('💡 Tip: Run debugDataService() in console for debug info');