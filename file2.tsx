import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { GitBranch, AlertCircle, CheckCircle, Clock, FileText, Settings, ExternalLink, Play, RefreshCw, Copy, Trash2, X, GitCommit } from 'lucide-react';
import { makeAuthenticatedRequest } from '../../services/auth';
import { showToast } from '../ui/Toast';

interface RepositoryConfig {
  id: number;
  project_id: number;
  repository_url: string;
  repository_name: string;
  provider: string;
  auto_analyze_prs: boolean;
  auto_create_test_cases: boolean;
  branch_patterns?: any;
  settings?: any;
  is_active: boolean;
  webhook_secret?: string;
  created_at: string;
  updated_at?: string;
}

interface PRAnalysis {
  id: number;
  repository_config_id: number;
  pr_number: number;
  pr_title: string;
  pr_description?: string;
  pr_url?: string;
  source_branch?: string;
  target_branch?: string;
  author?: string;
  event_type?: string;
  status: 'pending' | 'analyzing' | 'completed' | 'failed';
  lines_added: number;
  lines_removed: number;
  files_changed?: string[];
  analysis_results?: {
    commits_analyzed?: number;
    commits_info?: Array<{
      sha: string;
      short_sha: string;
      message: string;
      author: string;
      date: string;
    }>;
    base_ref?: string;
    head_ref?: string;
    analysis_range?: string;
  };
  ai_analysis_results?: {
    riskAssessment: string;
    recommendations: string[];
    affectedTestCases: number[];
    newTestScenariosNeeded: string[];
    testCaseDeltas: any[];
  };
  processed_at?: string;
  created_at: string;
}

interface TestCaseDelta {
  id: number;
  pull_request_analysis_id: number;
  test_case_id?: number;
  delta_type: 'create' | 'update' | 'affect';
  suggested_title?: string;
  suggested_preconditions?: string;
  suggested_steps?: string;
  suggested_expected_results?: string;
  suggested_priority?: string;
  suggested_type?: string;
  suggested_suite_id?: number;
  reasoning?: string;
  confidence: number;
  is_applied: boolean;
  applied_at?: string;
  applied_by_id?: number;
  created_at: string;
}

interface IntegrationLog {
  id: number;
  repository_config_id?: number;
  pull_request_analysis_id?: number;
  event_type: string;
  event_data?: any;
  success: boolean;
  error_message?: string;
  processing_time?: number;
  created_at: string;
}

interface GitHubIntegrationDashboardProps {
  projectId: number;
}

export function GitHubIntegrationDashboard({ projectId }: GitHubIntegrationDashboardProps) {
  const [repositories, setRepositories] = useState<RepositoryConfig[]>([]);
  const [showConnectModal, setShowConnectModal] = useState(false);
  const [connectForm, setConnectForm] = useState({
    repository_url: '',
    repository_name: '',
    provider: 'github',
    personal_access_token: '',
    auto_analyze_prs: true,
    auto_create_test_cases: true,
    branch_patterns: '',
  });
  const [connecting, setConnecting] = useState(false);
  const [prAnalyses, setPrAnalyses] = useState<PRAnalysis[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<PRAnalysis | null>(null);
  const [testDeltas, setTestDeltas] = useState<TestCaseDelta[]>([]);
  const [integrationLogs, setIntegrationLogs] = useState<IntegrationLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('repositories');
  const [refreshing, setRefreshing] = useState(false);
  const [selectedDeltaIds, setSelectedDeltaIds] = useState<Set<number>>(new Set());
  const [bulkApplying, setBulkApplying] = useState(false);
  const [triggeringAnalysis, setTriggeringAnalysis] = useState(false);
  
  // Ngrok URL from environment variables
  const ngrokUrl = import.meta.env.VITE_NGROK_URL || 'https://your-ngrok-url.ngrok.io';
  
  // Configuration dialog state
  const [showConfigDialog, setShowConfigDialog] = useState(false);
  const [configForm, setConfigForm] = useState<RepositoryConfig | null>(null);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    loadData();
  }, [projectId]);

  useEffect(() => {
    if (selectedAnalysis) {
      loadTestDeltas(selectedAnalysis.id);
    }
  }, [selectedAnalysis]);


  const handleConnectRepository = async () => {
    // Validation
    if (!connectForm.repository_url.trim()) {
      showToast('error', 'Repository URL is required');
      return;
    }
    if (!connectForm.repository_name.trim()) {
      showToast('error', 'Repository name is required');
      return;
    }
    if (!connectForm.personal_access_token.trim()) {
      showToast('error', 'Personal access token is required');
      return;
    }

    setConnecting(true);
    try {
      const payload = {
        project_id: projectId,
        repository_url: connectForm.repository_url.trim(),
        repository_name: connectForm.repository_name.trim(),
        provider: connectForm.provider,
        access_token: connectForm.personal_access_token.trim(),
        auto_analyze_prs: connectForm.auto_analyze_prs,
        auto_create_test_cases: connectForm.auto_create_test_cases,
        branch_patterns: {
          patterns: connectForm.branch_patterns
            ? connectForm.branch_patterns.split(',').map(p => p.trim())
            : []
        },
        // webhook_secret will be auto-generated by backend if not provided
      };

      console.log('Sending payload:', payload); // Debug log

      const response = await makeAuthenticatedRequest(
        '/api/v1/repositories/configs/',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }
      );
      if (response.ok) {
        const newRepo = await response.json();
        showToast('success', 'Repository connected successfully');
        setShowConnectModal(false);
        setConnectForm({
          repository_url: '',
          repository_name: '',
          provider: 'github',
          personal_access_token: '',
          auto_analyze_prs: true,
          auto_create_test_cases: true,
          branch_patterns: '',
        });
        
        // Fetch webhook details for the newly created repository
        try {
          const webhookResponse = await makeAuthenticatedRequest(
            `/api/v1/repositories/${newRepo.id}/webhook-details`
          );
          if (webhookResponse.ok) {
            const webhookData = await webhookResponse.json();
            // Update the repository with webhook details
            setRepositories(prev => prev.map(repo => 
              repo.id === newRepo.id 
                ? { ...repo, webhook_secret: webhookData.webhook_secret }
                : repo
            ));
          }
        } catch (error) {
          console.error('Failed to fetch webhook details:', error);
        }
        
        loadData();
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to connect repository');
      }
    } catch (error) {
      showToast('error', 'Failed to connect repository');
    } finally {
      setConnecting(false);
    }
  };

  const loadData = async () => {
    try {
      setLoading(true);
      // Load repositories
      const reposResponse = await makeAuthenticatedRequest(
        `/api/v1/repositories/configs/?project_id=${projectId}`
      );
      if (reposResponse.ok) {
        const reposData = await reposResponse.json();
        setRepositories(reposData);
      }
      // Load PR analyses
      const prResponse = await makeAuthenticatedRequest(
        `/api/v1/repositories/pr-analysis/?project_id=${projectId}&limit=20`
      );
      if (prResponse.ok) {
        const prData = await prResponse.json();
        setPrAnalyses(prData);
      }
      // Load integration logs
      const logsResponse = await makeAuthenticatedRequest(
        `/api/v1/repositories/integration-logs?limit=50`
      );
      if (logsResponse.ok) {
        const logsData = await logsResponse.json();
        setIntegrationLogs(logsData);
      }
    } catch (error) {
      console.error('Failed to load integration data:', error);
      showToast('error', 'Failed to load integration data');
    } finally {
      setLoading(false);
    }
  };

  const triggerManualAnalysis = async () => {
    if (repositories.length === 0) {
      showToast('error', 'No repositories configured');
      return;
    }

    // Use the first active repository for analysis
    const activeRepo = repositories.find(repo => repo.is_active);
    if (!activeRepo) {
      showToast('error', 'No active repositories found');
      return;
    }

    setTriggeringAnalysis(true);
    try {
      const response = await makeAuthenticatedRequest(
        '/api/v1/ai-analysis/trigger-latest-commit',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            repository_config_id: activeRepo.id,
            branch: 'main'
          })
        }
      );

      if (response.ok) {
        const result = await response.json();
        showToast('success', `🤖 AI analysis started! Analysis ID: ${result.analysis_id}`);
        
        // Refresh data to show the new analysis
        setTimeout(() => {
          loadData();
        }, 1000);
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to trigger AI analysis');
      }
    } catch (error) {
      console.error('Failed to trigger AI analysis:', error);
      showToast('error', 'Failed to trigger AI analysis');
    } finally {
      setTriggeringAnalysis(false);
    }
  };

  // Function to trigger AI analysis for a specific PR/commit record
  const triggerAnalysisForRecord = async (analysisId: number, title: string) => {
    setTriggeringAnalysis(true);
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/ai-analysis/trigger-analysis/${analysisId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (response.ok) {
        const result = await response.json();
        showToast('success', `🤖 AI analysis started for: ${title}`);
        
        // Refresh data to show the updated status
        setTimeout(() => {
          loadData();
        }, 1000);
        
        // Refresh page after successful analysis to ensure updated data is displayed
        setTimeout(() => {
          window.location.reload();
        }, 3000);
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to trigger AI analysis');
      }
    } catch (error) {
      console.error('Failed to trigger analysis:', error);
      showToast('error', 'Failed to trigger AI analysis');
    } finally {
      setTriggeringAnalysis(false);
    }
  };

  const refreshData = async () => {
    setRefreshing(true);
    try {
      // Simply reload all data from database - webhooks will have already created records
      await loadData();
      showToast('success', 'Data refreshed from database');
    } catch (error) {
      console.error('Failed to refresh:', error);
      showToast('error', 'Failed to refresh data');
    } finally {
      setRefreshing(false);
    }
  };

  // Manual discovery function for users who want to fetch from GitHub API
  const discoverFromGitHub = async () => {
    setRefreshing(true);
    try {
      let totalNewPRs = 0;
      let totalNewCommits = 0;
      
      // Fetch from GitHub API for each active repository
      for (const repo of repositories.filter(r => r.is_active)) {
        try {
          const fetchResponse = await makeAuthenticatedRequest(
            `/api/v1/webhooks/repositories/${repo.id}/fetch-prs`,
            { method: 'GET' }
          );
          if (fetchResponse.ok) {
            const fetchData = await fetchResponse.json();
            // For the new GET endpoint, we just get existing data
            console.log(`Repository ${repo.repository_name}:`, fetchData);
          }
        } catch (error) {
          console.error(`Failed to fetch data for ${repo.repository_name}:`, error);
        }
      }
      
      // Reload data to show any updates
      await loadData();
      showToast('success', 'Data synchronized with database');
    } catch (error) {
      console.error('Failed to discover:', error);
      showToast('error', 'Failed to discover data');
    } finally {
      setRefreshing(false);
    }
  };



  const loadTestDeltas = async (analysisId: number) => {
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/pr-analyses/${analysisId}/deltas`
      );
      if (response.ok) {
        const data = await response.json();
        setTestDeltas(data);
      }
    } catch (error) {
      console.error('Failed to load test deltas:', error);
    }
  };

  const applyTestDelta = async (deltaId: number) => {
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/test-case-deltas/${deltaId}/apply`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        const result = await response.json();
        console.log('Apply result:', result); // Debug log
        showToast('success', `✅ AI recommendation applied! Test case #${result.test_case_id || 'created'}`);
        // Reload test deltas
        if (selectedAnalysis) {
          loadTestDeltas(selectedAnalysis.id);
        }
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to apply test delta');
      }
    } catch (error) {
      console.error('Failed to apply test delta:', error);
      showToast('error', 'Failed to apply test delta');
    }
  };

  const dismissTestDelta = async (deltaId: number, reason?: string) => {
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/test-case-deltas/${deltaId}/dismiss`,
        { 
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: reason || 'Dismissed by user' })
        }
      );
      
      if (response.ok) {
        showToast('success', '🗑️ AI recommendation dismissed');
        // Reload test deltas
        if (selectedAnalysis) {
          loadTestDeltas(selectedAnalysis.id);
        }
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to dismiss test delta');
      }
    } catch (error) {
      console.error('Failed to dismiss test delta:', error);
      showToast('error', 'Failed to dismiss test delta');
    }
  };

  const bulkApplyDeltas = async (deltaIds: number[]) => {
    if (deltaIds.length === 0) {
      showToast('error', 'No deltas selected');
      return;
    }

    setBulkApplying(true);
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/test-case-deltas/bulk-apply`,
        { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(deltaIds)  // Backend expects array directly
        }
      );
      
      if (response.ok) {
        const result = await response.json();
        console.log('Bulk apply result:', result); // Debug log
        
        const appliedCount = result.applied?.length || 0;
        const failedCount = result.failed?.length || 0;
        
        if (appliedCount > 0) {
          showToast('success', `🎉 ${appliedCount} AI recommendation${appliedCount > 1 ? 's' : ''} applied successfully!`);
        }
        
        if (failedCount > 0) {
          showToast('warning', `⚠️ ${failedCount} recommendation${failedCount > 1 ? 's' : ''} failed to apply`);
        }
        
        if (appliedCount === 0 && failedCount === 0) {
          showToast('info', '✅ Bulk apply completed');
        }
        
        // Clear selections and reload test deltas
        setSelectedDeltaIds(new Set());
        if (selectedAnalysis) {
          loadTestDeltas(selectedAnalysis.id);
        }
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to bulk apply test deltas');
      }
    } catch (error) {
      console.error('Failed to bulk apply test deltas:', error);
      showToast('error', 'Failed to bulk apply test deltas');
    } finally {
      setBulkApplying(false);
    }
  };

  const handleDeltaSelection = (deltaId: number, isSelected: boolean) => {
    const newSelection = new Set(selectedDeltaIds);
    if (isSelected) {
      newSelection.add(deltaId);
    } else {
      newSelection.delete(deltaId);
    }
    setSelectedDeltaIds(newSelection);
  };

  const handleSelectAllDeltas = (selectAll: boolean) => {
    if (selectAll) {
      const applicableIds = testDeltas
        .filter(delta => !delta.is_applied && delta.confidence > 0)
        .map(delta => delta.id);
      setSelectedDeltaIds(new Set(applicableIds));
    } else {
      setSelectedDeltaIds(new Set());
    }
  };

  const triggerAnalysis = async (analysisId: number) => {
    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/pr-analyses/${analysisId}/trigger`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        showToast('success', 'Analysis triggered successfully');
        loadData();
        // Refresh page after successful analysis to ensure updated data is displayed
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to trigger analysis');
      }
    } catch (error) {
      console.error('Failed to trigger analysis:', error);
      showToast('error', 'Failed to trigger analysis');
    }
  };

  const handleDeleteRepository = async (repositoryId: number) => {
    if (!confirm('Are you sure you want to delete this repository configuration? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/configs/${repositoryId}/`,
        { method: 'DELETE' }
      );
      
      if (response.ok) {
        showToast('success', 'Repository deleted successfully');
        loadData(); // Reload the repository list
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to delete repository');
      }
    } catch (error) {
      console.error('Failed to delete repository:', error);
      showToast('error', 'Failed to delete repository');
    }
  };

  const handleOpenConfigDialog = (repo: RepositoryConfig) => {
    // Convert branch_patterns from object/array to string for display
    let branchPatternsStr = '';
    if (repo.branch_patterns) {
      if (typeof repo.branch_patterns === 'string') {
        branchPatternsStr = repo.branch_patterns;
      } else if (Array.isArray(repo.branch_patterns)) {
        branchPatternsStr = repo.branch_patterns.join(', ');
      } else if (typeof repo.branch_patterns === 'object' && repo.branch_patterns.patterns) {
        branchPatternsStr = Array.isArray(repo.branch_patterns.patterns) 
          ? repo.branch_patterns.patterns.join(', ')
          : String(repo.branch_patterns.patterns);
      }
    }
    setConfigForm({ ...repo, branch_patterns: branchPatternsStr as any });
    setShowConfigDialog(true);
  };

  const handleUpdateConfiguration = async () => {
    if (!configForm) return;

    setUpdating(true);
    try {
      // Parse branch_patterns string back to array/object format
      let branchPatternsData: any = null;
      if (configForm.branch_patterns && typeof configForm.branch_patterns === 'string') {
        const patterns = configForm.branch_patterns
          .split(',')
          .map(p => p.trim())
          .filter(p => p.length > 0);
        branchPatternsData = { patterns };
      } else {
        branchPatternsData = configForm.branch_patterns;
      }

      const response = await makeAuthenticatedRequest(
        `/api/v1/repositories/configs/${configForm.id}/`,
        {
          method: 'PUT',
          body: JSON.stringify({
            auto_analyze_prs: configForm.auto_analyze_prs,
            auto_create_test_cases: configForm.auto_create_test_cases,
            branch_patterns: branchPatternsData,
            is_active: configForm.is_active,
          }),
        }
      );

      if (response.ok) {
        showToast('success', 'Configuration updated successfully');
        setShowConfigDialog(false);
        loadData(); // Reload the repository list
      } else {
        const error = await response.json();
        showToast('error', error.detail || 'Failed to update configuration');
      }
    } catch (error) {
      console.error('Failed to update configuration:', error);
      showToast('error', 'Failed to update configuration');
    } finally {
      setUpdating(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      pending: { color: 'bg-yellow-100 text-yellow-800 border-yellow-200', icon: Clock, label: 'Pending' },
      analyzing: { color: 'bg-blue-100 text-blue-800 border-blue-200', icon: RefreshCw, label: 'Processing' },
      completed: { color: 'bg-green-100 text-green-800 border-green-200', icon: CheckCircle, label: 'Completed' },
      failed: { color: 'bg-red-100 text-red-800 border-red-200', icon: AlertCircle, label: 'Failed' }
    };
    
    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.pending;
    const Icon = config.icon;
    
    // Only show spinner and "auto-updating" for actively analyzing records, not pending ones
    const isAnalyzing = status === 'analyzing';
    const showSpinner = isAnalyzing;
    
    return (
      <Badge className={`${config.color} border flex items-center gap-1`}>
        <Icon className={`w-3 h-3 ${showSpinner ? 'animate-spin' : ''}`} />
        {config.label}
        {showSpinner && <span className="text-xs">(auto-updating)</span>}
      </Badge>
    );
  };

  const getRiskBadge = (risk: string) => {
    const riskConfig = {
      low: 'bg-green-100 text-green-800 border-green-200',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      high: 'bg-orange-100 text-orange-800 border-orange-200',
      critical: 'bg-red-100 text-red-800 border-red-200'
    };
    
    return (
      <Badge className={`${riskConfig[risk as keyof typeof riskConfig] || riskConfig.medium} border`}>
        {risk.toUpperCase()}
      </Badge>
    );
  };

  const getDeltaTypeBadge = (deltaType: string) => {
    const typeConfig = {
      create: 'bg-blue-100 text-blue-800 border-blue-200',
      update: 'bg-orange-100 text-orange-800 border-orange-200',
      affect: 'bg-purple-100 text-purple-800 border-purple-200'
    };
    
    return (
      <Badge className={`${typeConfig[deltaType as keyof typeof typeConfig] || typeConfig.create} border`}>
        {deltaType.toUpperCase()}
      </Badge>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">GitHub Integration</h2>
        <div className="flex items-center gap-2">
          <Button 
            onClick={refreshData} 
            variant="outline"
            disabled={refreshing}
            className="flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
          <Button 
            onClick={triggerManualAnalysis}
            disabled={triggeringAnalysis}
            variant="outline"
            className="flex items-center gap-2"
          >
            <Play className={`w-4 h-4 ${triggeringAnalysis ? 'animate-spin' : ''}`} />
            {triggeringAnalysis ? 'Analyzing...' : 'Analyze Latest Commit'}
          </Button>
        </div>
      </div>

      {/* Connect Repository Modal */}
      {showConnectModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-bold mb-4">Connect GitHub Repository</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Repository URL <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  className="w-full border rounded px-3 py-2"
                  placeholder="https://github.com/user/repo"
                  value={connectForm.repository_url}
                  onChange={e => setConnectForm({ ...connectForm, repository_url: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Repository Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  className="w-full border rounded px-3 py-2"
                  placeholder="user/repo"
                  value={connectForm.repository_name}
                  onChange={e => setConnectForm({ ...connectForm, repository_name: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Personal Access Token <span className="text-red-500">*</span>
                </label>
                <input
                  type="password"
                  className="w-full border rounded px-3 py-2"
                  placeholder="ghp_..."
                  value={connectForm.personal_access_token}
                  onChange={e => setConnectForm({ ...connectForm, personal_access_token: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Branch Patterns (optional)
                </label>
                <input
                  type="text"
                  className="w-full border rounded px-3 py-2"
                  placeholder="main, develop, feature/*"
                  value={connectForm.branch_patterns}
                  onChange={e => setConnectForm({ ...connectForm, branch_patterns: e.target.value })}
                />
                <p className="text-xs text-gray-500 mt-1">Comma-separated list of branches to monitor</p>
              </div>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={connectForm.auto_analyze_prs}
                    onChange={e => setConnectForm({ ...connectForm, auto_analyze_prs: e.target.checked })}
                  /> Auto-analyze PRs
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={connectForm.auto_create_test_cases}
                    onChange={e => setConnectForm({ ...connectForm, auto_create_test_cases: e.target.checked })}
                  /> Auto-create tests
                </label>
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={() => setShowConnectModal(false)} disabled={connecting}>Cancel</Button>
              <Button onClick={handleConnectRepository} disabled={connecting}>
                {connecting ? 'Connecting...' : 'Connect'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Configuration Dialog */}
      {showConfigDialog && configForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-bold mb-4">Configure Repository</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Repository: <span className="font-semibold">{configForm.repository_name}</span>
                </label>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Branch Patterns
                </label>
                <input
                  type="text"
                  className="w-full border rounded px-3 py-2"
                  placeholder="main, develop, feature/*"
                  value={configForm.branch_patterns || ''}
                  onChange={e => setConfigForm({ ...configForm, branch_patterns: e.target.value })}
                />
                <p className="text-xs text-gray-500 mt-1">Comma-separated list of branches to monitor</p>
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={configForm.auto_analyze_prs}
                    onChange={e => setConfigForm({ ...configForm, auto_analyze_prs: e.target.checked })}
                  />
                  <span>Auto-analyze Pull Requests</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={configForm.auto_create_test_cases}
                    onChange={e => setConfigForm({ ...configForm, auto_create_test_cases: e.target.checked })}
                  />
                  <span>Auto-create Test Cases</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={configForm.is_active}
                    onChange={e => setConfigForm({ ...configForm, is_active: e.target.checked })}
                  />
                  <span>Active</span>
                </label>
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={() => setShowConfigDialog(false)} disabled={updating}>
                Cancel
              </Button>
              <Button onClick={handleUpdateConfiguration} disabled={updating}>
                {updating ? 'Updating...' : 'Update Configuration'}
              </Button>
            </div>
          </div>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="repositories">Repositories ({repositories.length})</TabsTrigger>
          <TabsTrigger value="analyses">PR Analyses ({prAnalyses.length})</TabsTrigger>
          <TabsTrigger value="logs">Integration Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="repositories" className="space-y-4">
          <div className="grid gap-4">
            {repositories.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center text-gray-500">
                    <GitBranch className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p className="text-lg mb-2">No repositories configured</p>
                    <p className="text-sm">Connect your GitHub or GitLab repositories to start receiving automated test recommendations.</p>
                    <Button className="mt-4" onClick={() => setShowConnectModal(true)}>
                      Connect Repository
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              repositories.map((repo) => (
                <Card key={repo.id} className="hover:shadow-md transition-shadow">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-lg font-medium flex items-center gap-2">
                      <GitBranch className="w-5 h-5 text-gray-600" />
                      {repo.repository_name || 'Unknown Repository'}
                    </CardTitle>
                    <Badge className={repo.is_active ? 'bg-green-100 text-green-800 border-green-200' : 'bg-gray-100 text-gray-800 border-gray-200'}>
                      {repo.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <ExternalLink className="w-4 h-4" />
                        <a href={repo.repository_url} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600">
                          {repo.repository_url}
                        </a>
                      </div>
                      
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline" className="text-xs">
                          {repo.provider.toUpperCase()}
                        </Badge>
                        {repo.auto_analyze_prs && (
                          <Badge variant="outline" className="text-xs">Auto-analyze PRs</Badge>
                        )}
                        {repo.auto_create_test_cases && (
                          <Badge variant="outline" className="text-xs">Auto-create tests</Badge>
                        )}
                      </div>
                      
                      
                      {/* Webhook Information */}
                      <div className="p-2 bg-gray-50 rounded border">
                        <div className="text-xs font-medium text-gray-700 mb-1">Webhook Secret:</div>
                        <div className="flex items-center gap-2">
                          <code className="text-xs bg-white px-2 py-1 rounded border font-mono flex-1">
                            {repo.webhook_secret === '***masked***' ? '***masked***' : repo.webhook_secret}
                          </code>
                          {repo.webhook_secret === '***masked***' ? (
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 text-xs px-2"
                              onClick={async () => {
                                try {
                                  const response = await makeAuthenticatedRequest(
                                    `/api/v1/repositories/${repo.id}/webhook-details`
                                  );
                                  if (response.ok) {
                                    const data = await response.json();
                                    setRepositories(prev => prev.map(r => 
                                      r.id === repo.id 
                                        ? { ...r, webhook_secret: data.webhook_secret, repository_name: data.repository_name }
                                        : r
                                    ));
                                    showToast('success', 'Webhook secret revealed');
                                  }
                                } catch (error) {
                                  showToast('error', 'Failed to reveal webhook secret');
                                }
                              }}
                            >
                              Reveal
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={() => {
                                navigator.clipboard.writeText(repo.webhook_secret);
                                showToast('success', 'Webhook secret copied to clipboard');
                              }}
                            >
                              <Copy className="w-3 h-3" />
                            </Button>
                          )}
                        </div>
                      </div>

                      {/* Webhook URL Information */}
                      <div className="p-2 bg-blue-50 rounded border">
                        <div className="text-xs font-medium text-blue-700 mb-1">
                          GitHub Webhook URL:
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="text-xs bg-white px-2 py-1 rounded border font-mono flex-1 break-all">
                            {`${ngrokUrl}/api/v1/webhooks/github/${repo.id}`}
                          </code>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 w-6 p-0"
                            onClick={() => {
                              const webhookUrl = `${ngrokUrl}/api/v1/webhooks/github/${repo.id}`;
                              navigator.clipboard.writeText(webhookUrl);
                              showToast('success', 'Webhook URL copied to clipboard');
                            }}
                          >
                            <Copy className="w-3 h-3" />
                          </Button>
                        </div>
                        <div className="text-xs text-blue-600 mt-1">
                          Copy this URL to GitHub repository Settings → Webhooks → Add webhook
                        </div>
                      </div>

                      <div className="text-xs text-gray-500">
                        Created {new Date(repo.created_at).toLocaleDateString()}
                      </div>
                      
                      {/* Action Buttons */}
                      <div className="flex gap-2 pt-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleOpenConfigDialog(repo)}
                        >
                          <Settings className="w-3 h-3 mr-1" />
                          Configure
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleDeleteRepository(repo.id)}
                        >
                          <Trash2 className="w-3 h-3 mr-1" />
                          Delete
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="analyses" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            {/* PR Analyses List */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Pull Request Analyses</h3>
              {prAnalyses.length === 0 ? (
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-center text-gray-500">
                      <FileText className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                      <p>No PR analyses found</p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                prAnalyses.map((analysis) => (
                  <Card 
                    key={analysis.id} 
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedAnalysis?.id === analysis.id ? 'ring-2 ring-blue-500 shadow-md' : ''
                    }`}
                    onClick={() => setSelectedAnalysis(analysis)}
                  >
                    <CardHeader className="pb-2">
                      <div className="flex items-start justify-between">
                        <CardTitle className="text-sm font-medium line-clamp-2">
                          PR #{analysis.pr_number}: {analysis.pr_title}
                        </CardTitle>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(analysis.status)}
                          {analysis.status === 'pending' && (
                            <Button 
                              size="sm" 
                              variant="outline" 
                              className="h-6 px-2 text-xs"
                              onClick={(e) => {
                                e.stopPropagation(); // Prevent card selection
                                triggerAnalysisForRecord(analysis.id, analysis.pr_title);
                              }}
                              disabled={triggeringAnalysis}
                            >
                              🤖 Analyze
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs text-gray-600">
                          <GitBranch className="w-3 h-3" />
                          <span>{analysis.source_branch} → {analysis.target_branch}</span>
                        </div>
                        
                        <div className="flex items-center justify-between text-xs text-gray-600">
                          <span>By {analysis.author}</span>
                          <span>{analysis.files_changed?.length || 0} files</span>
                        </div>
                        
                        <div className="flex items-center gap-4 text-xs text-gray-600">
                          <span className="text-green-600">+{analysis.lines_added}</span>
                          <span className="text-red-600">-{analysis.lines_removed}</span>
                        </div>
                        
                        {analysis.ai_analysis_results?.riskAssessment && (
                          <div className="mt-2">
                            {getRiskBadge(analysis.ai_analysis_results.riskAssessment)}
                          </div>
                        )}
                        
                        <div className="text-xs text-gray-500">
                          {new Date(analysis.created_at).toLocaleString()}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>

            {/* Analysis Details */}
            <div className="space-y-4">
              {selectedAnalysis ? (
                <>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium">Analysis Details</h3>
                    <div className="flex gap-2">
                      {selectedAnalysis.pr_url && (
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => window.open(selectedAnalysis.pr_url, '_blank')}
                          className="flex items-center gap-1"
                        >
                          <ExternalLink className="w-3 h-3" />
                          View PR
                        </Button>
                      )}
                      {(selectedAnalysis.status === 'failed' || selectedAnalysis.status === 'pending') && (
                        <Button 
                          size="sm" 
                          onClick={() => triggerAnalysis(selectedAnalysis.id)}
                          className="flex items-center gap-1"
                        >
                          <Play className="w-3 h-3" />
                          {selectedAnalysis.status === 'failed' ? 'Retry' : 'Analyze'}
                        </Button>
                      )}
                    </div>
                  </div>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <span>PR #{selectedAnalysis.pr_number}: {selectedAnalysis.pr_title}</span>
                        {getStatusBadge(selectedAnalysis.status)}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {selectedAnalysis.pr_description && (
                        <div>
                          <h4 className="text-sm font-medium mb-1">Description</h4>
                          <p className="text-sm text-gray-600 whitespace-pre-wrap">
                            {selectedAnalysis.pr_description}
                          </p>
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Author:</span> {selectedAnalysis.author}
                        </div>
                        <div>
                          <span className="font-medium">Files:</span> {selectedAnalysis.files_changed?.length || 0}
                        </div>
                        <div>
                          <span className="font-medium">Added:</span> 
                          <span className="text-green-600 ml-1">+{selectedAnalysis.lines_added}</span>
                        </div>
                        <div>
                          <span className="font-medium">Removed:</span>
                          <span className="text-red-600 ml-1">-{selectedAnalysis.lines_removed}</span>
                        </div>
                      </div>

                      {/* Commit Analysis Info */}
                      {selectedAnalysis.analysis_results?.commits_info && selectedAnalysis.analysis_results.commits_info.length > 0 && (
                        <div className="border-t pt-4">
                          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                            <GitCommit className="w-4 h-4" />
                            Commits Analyzed ({selectedAnalysis.analysis_results.commits_analyzed})
                          </h4>
                          {selectedAnalysis.analysis_results.analysis_range && (
                            <p className="text-xs text-gray-500 mb-2">
                              Range: {selectedAnalysis.analysis_results.analysis_range}
                            </p>
                          )}
                          <div className="space-y-2 max-h-32 overflow-y-auto">
                            {selectedAnalysis.analysis_results.commits_info.map((commit: any, index: number) => (
                              <div key={commit.sha} className="flex items-start gap-2 text-sm p-2 bg-gray-50 rounded">
                                <code className="text-xs bg-gray-200 px-1 rounded font-mono">
                                  {commit.short_sha}
                                </code>
                                <div className="flex-1 min-w-0">
                                  <p className="truncate font-medium">{commit.message}</p>
                                  <p className="text-xs text-gray-500">by {commit.author}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedAnalysis.ai_analysis_results && (
                        <div className="space-y-3 border-t pt-4">
                          <div>
                            <h4 className="text-sm font-medium mb-2">AI Analysis Results</h4>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-sm font-medium">Risk Assessment:</span>
                              {getRiskBadge(selectedAnalysis.ai_analysis_results.riskAssessment)}
                            </div>
                          </div>

                          {selectedAnalysis.ai_analysis_results.recommendations && 
                           selectedAnalysis.ai_analysis_results.recommendations.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium mb-1">Recommendations</h4>
                              <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                                {selectedAnalysis.ai_analysis_results.recommendations.map((rec: string, index: number) => (
                                  <li key={index}>{rec}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {selectedAnalysis.ai_analysis_results.affectedTestCases?.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium mb-1">Affected Test Cases</h4>
                              <p className="text-sm text-gray-600">
                                {selectedAnalysis.ai_analysis_results.affectedTestCases.length} test cases may be impacted
                              </p>
                            </div>
                          )}

                          {selectedAnalysis.ai_analysis_results.newTestScenariosNeeded?.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium mb-1">New Test Scenarios Needed</h4>
                              <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                                {selectedAnalysis.ai_analysis_results.newTestScenariosNeeded.map((scenario: string, index: number) => (
                                  <li key={index}>{scenario}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Test Deltas */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-base font-medium">Test Recommendations ({testDeltas.length})</h4>
                      {testDeltas.length > 0 && (
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleSelectAllDeltas(selectedDeltaIds.size === 0)}
                            className="text-xs"
                          >
                            {selectedDeltaIds.size === 0 ? 'Select All' : 'Deselect All'}
                          </Button>
                          {selectedDeltaIds.size > 0 && (
                            <Button
                              size="sm"
                              onClick={() => bulkApplyDeltas(Array.from(selectedDeltaIds))}
                              disabled={bulkApplying}
                              className="bg-green-600 hover:bg-green-700 text-xs"
                            >
                              {bulkApplying ? 'Applying...' : `Apply ${selectedDeltaIds.size} Selected`}
                            </Button>
                          )}
                        </div>
                      )}
                    </div>
                    {testDeltas.length === 0 ? (
                      <Card>
                        <CardContent className="pt-4">
                          <p className="text-center text-gray-500 text-sm">
                            No test recommendations available
                          </p>
                        </CardContent>
                      </Card>
                    ) : (
                      testDeltas.map((delta) => (
                        <Card key={delta.id} className="border-l-4 border-l-blue-500">
                          <CardContent className="pt-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-2">
                                {/* Checkbox for bulk selection */}
                                {!delta.is_applied && delta.confidence > 0 && (
                                  <input
                                    type="checkbox"
                                    checked={selectedDeltaIds.has(delta.id)}
                                    onChange={(e) => handleDeltaSelection(delta.id, e.target.checked)}
                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                  />
                                )}
                                
                                {getDeltaTypeBadge(delta.delta_type)}
                                <Badge className={`border ${delta.confidence < 0 ? 'bg-gray-100 text-gray-800 border-gray-200' : 'bg-blue-100 text-blue-800 border-blue-200'}`}>
                                  {delta.confidence < 0 ? 'Dismissed' : `${delta.confidence}% confidence`}
                                </Badge>
                                {delta.suggested_priority && (
                                  <Badge variant="outline" className="text-xs">
                                    {delta.suggested_priority} priority
                                  </Badge>
                                )}
                              </div>
                              
                              {/* Action buttons */}
                              <div className="flex items-center gap-1">
                                {!delta.is_applied && delta.confidence > 0 ? (
                                  <>
                                    <Button 
                                      size="sm" 
                                      onClick={() => applyTestDelta(delta.id)}
                                      className="flex items-center gap-1"
                                    >
                                      <CheckCircle className="w-3 h-3" />
                                      Apply
                                    </Button>
                                    <Button 
                                      size="sm"
                                      variant="outline"
                                      onClick={() => dismissTestDelta(delta.id)}
                                      className="flex items-center gap-1 text-gray-600 hover:text-red-600"
                                    >
                                      <X className="w-3 h-3" />
                                      Dismiss
                                    </Button>
                                  </>
                                ) : delta.confidence < 0 ? (
                                  <Badge className="bg-gray-100 text-gray-800 border-gray-200">
                                    <X className="w-3 h-3 mr-1" />
                                    Dismissed
                                  </Badge>
                                ) : (
                                  <Badge className="bg-green-100 text-green-800 border-green-200">
                                    <CheckCircle className="w-3 h-3 mr-1" />
                                    Applied
                                  </Badge>
                                )}
                              </div>
                            </div>

                            {delta.suggested_title && (
                              <div className="mb-2">
                                <h5 className="text-sm font-medium">{delta.suggested_title}</h5>
                              </div>
                            )}

                            <div className="text-sm text-gray-600 space-y-2">
                              {delta.reasoning && (
                                <div>
                                  <strong>Reasoning:</strong> {delta.reasoning}
                                </div>
                              )}
                              
                              {delta.suggested_steps && (
                                <div>
                                  <strong>Suggested Steps:</strong>
                                  <div className="mt-1 p-2 bg-gray-50 rounded text-xs">
                                    {delta.suggested_steps}
                                  </div>
                                </div>
                              )}

                              {delta.suggested_expected_results && (
                                <div>
                                  <strong>Expected Results:</strong>
                                  <div className="mt-1 p-2 bg-gray-50 rounded text-xs">
                                    {delta.suggested_expected_results}
                                  </div>
                                </div>
                              )}

                              <div className="flex gap-4 text-xs">
                                {delta.suggested_type && (
                                  <span><strong>Type:</strong> {delta.suggested_type}</span>
                                )}
                                {delta.is_applied && delta.applied_at && (
                                  <span><strong>Applied:</strong> {new Date(delta.applied_at).toLocaleString()}</span>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-center text-gray-500">
                      <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                      <p className="text-lg mb-2">Select a PR analysis</p>
                      <p className="text-sm">Choose a pull request analysis from the list to view detailed results and test recommendations.</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="logs" className="space-y-4">
          <h3 className="text-lg font-medium">Integration Logs</h3>
          <div className="space-y-2">
            {integrationLogs.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center text-gray-500">
                    <Settings className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                    <p>No integration logs found</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              integrationLogs.map((log) => (
                <Card key={log.id} className="hover:shadow-sm transition-shadow">
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge className={log.success ? 'bg-green-100 text-green-800 border-green-200' : 'bg-red-100 text-red-800 border-red-200'}>
                          {log.success ? 'Success' : 'Failed'}
                        </Badge>
                        <span className="text-sm font-medium">{log.event_type}</span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(log.created_at).toLocaleString()}
                      </span>
                    </div>
                    
                    {log.error_message && (
                      <p className="text-sm text-red-600 mb-2 p-2 bg-red-50 rounded">
                        {log.error_message}
                      </p>
                    )}
                    
                    {log.event_data && (
                      <div className="text-xs text-gray-600 mb-2">
                        <details>
                          <summary className="cursor-pointer hover:text-gray-800">Event Data</summary>
                          <pre className="mt-1 p-2 bg-gray-50 rounded overflow-auto">
                            {JSON.stringify(log.event_data, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                    
                    {log.processing_time && (
                      <p className="text-xs text-gray-600">
                        Processing time: {log.processing_time}ms
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
