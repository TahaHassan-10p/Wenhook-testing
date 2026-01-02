import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '../ui/dialog';
import { ArrowLeft, FileText, History, RotateCcw, Save, ChevronDown, ChevronRight } from 'lucide-react';
import { toast } from 'sonner';
import { AggregateVersionService, AggregateVersion } from '../../services/aggregate-versions';
import { projectService, type Project } from '../../services/projects';

interface AggregateViewerModuleProps {
  projectId: number;
  onNavigate: (module: string, data?: any) => void;
}

export default function AggregateViewerModule({ projectId, onNavigate }: AggregateViewerModuleProps) {
  const [project, setProject] = useState<Project | null>(null);
  const [versions, setVersions] = useState<AggregateVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [currentAggregate, setCurrentAggregate] = useState<any>(null);
  const [displayedAggregate, setDisplayedAggregate] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showRevertDialog, setShowRevertDialog] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveDescription, setSaveDescription] = useState('');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['milestones']));

  useEffect(() => {
    if (projectId) {
      loadData();
    }
  }, [projectId]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [projectData, versionsData] = await Promise.all([
        projectService.getProject(projectId),
        AggregateVersionService.getVersions(projectId),
      ]);
      
      setProject(projectData);
      setVersions(versionsData.versions);
      setCurrentAggregate(versionsData.current_aggregate);
      setDisplayedAggregate(versionsData.current_aggregate);
      setSelectedVersion(null); // Current version
    } catch (error) {
      console.error('Failed to load aggregate data:', error);
      toast.error('Failed to load aggregate data');
    } finally {
      setLoading(false);
    }
  };

  const handleVersionChange = (versionId: string) => {
    if (versionId === 'current') {
      setSelectedVersion(null);
      setDisplayedAggregate(currentAggregate);
    } else {
      const version = versions.find(v => v.id === parseInt(versionId));
      if (version) {
        setSelectedVersion(version.id);
        setDisplayedAggregate(version.aggregate_data);
      }
    }
  };

  const handleSaveVersion = async () => {
    try {
      await AggregateVersionService.createVersion(projectId, saveDescription || undefined);
      toast.success('Version saved successfully');
      setShowSaveDialog(false);
      setSaveDescription('');
      await loadData();
    } catch (error) {
      console.error('Failed to save version:', error);
      toast.error('Failed to save version');
    }
  };

  const handleRevert = async () => {
    if (!selectedVersion) return;
    
    try {
      await AggregateVersionService.revertToVersion(projectId, selectedVersion);
      toast.success('Reverted to selected version');
      setShowRevertDialog(false);
      await loadData();
    } catch (error) {
      console.error('Failed to revert:', error);
      toast.error('Failed to revert to version');
    }
  };

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const renderSection = (title: string, key: string, data: any) => {
    const isExpanded = expandedSections.has(key);
    const items = Array.isArray(data) ? data : [];
    
    return (
      <Card key={key} className="mb-4">
        <CardHeader 
          className="cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => toggleSection(key)}
        >
          <CardTitle className="text-lg flex items-center justify-between">
            <span className="flex items-center gap-2">
              {isExpanded ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
              {title}
              <span className="text-sm text-gray-500">({items.length})</span>
            </span>
          </CardTitle>
        </CardHeader>
        {isExpanded && (
          <CardContent>
            {items.length > 0 ? (
              <div className="space-y-2">
                {items.map((item: any, idx: number) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-sm font-medium text-gray-900 mb-1">
                      {item.name || item.title || `Item ${idx + 1}`}
                    </div>
                    {item.description && (
                      <div className="text-xs text-gray-600 mb-2">{item.description}</div>
                    )}
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap bg-white p-2 rounded border">
                      {JSON.stringify(item, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No {title.toLowerCase()} in this version</p>
            )}
          </CardContent>
        )}
      </Card>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading aggregate data...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 max-w-7xl">
      {/* Header */}
      <div className="mb-6">
        <Button
          variant="ghost"
          onClick={() => onNavigate('project-details', { projectId, activeTab: 'ai-info' })}
          className="mb-4"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Project
        </Button>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
              <FileText className="h-8 w-8 text-purple-600" />
              Aggregate Response Viewer
            </h1>
            <p className="text-gray-600 mt-1">{project?.name}</p>
          </div>
          
          <div className="flex items-center gap-2">
            <Button onClick={() => setShowSaveDialog(true)} variant="outline">
              <Save className="h-4 w-4 mr-2" />
              Save Version
            </Button>
            {selectedVersion && (
              <Button onClick={() => setShowRevertDialog(true)}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Revert to This Version
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Version Selector */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <History className="h-5 w-5" />
            Version History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700">Select Version:</label>
            <Select
              value={selectedVersion ? selectedVersion.toString() : 'current'}
              onValueChange={handleVersionChange}
            >
              <SelectTrigger className="w-96">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="current">
                  Current (Latest)
                </SelectItem>
                {versions.map(version => (
                  <SelectItem key={version.id} value={version.id.toString()}>
                    Version {version.version_number} - {new Date(version.created_at).toLocaleString()}
                    {version.description && ` - ${version.description}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedVersion && (
              <span className="text-sm text-gray-500">
                Viewing historical version (not currently active)
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Aggregate Content */}
      {displayedAggregate ? (
        <div className="space-y-4">
          {displayedAggregate.milestones && renderSection('Milestones', 'milestones', displayedAggregate.milestones)}
          {displayedAggregate.test_plans && renderSection('Test Plans', 'test_plans', displayedAggregate.test_plans)}
          {displayedAggregate.test_suites && renderSection('Test Suites', 'test_suites', displayedAggregate.test_suites)}
          {displayedAggregate.test_cases && renderSection('Test Cases', 'test_cases', displayedAggregate.test_cases)}
          {displayedAggregate.test_runs && renderSection('Test Runs', 'test_runs', displayedAggregate.test_runs)}
          
          {/* Full JSON View */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Full JSON Data</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto max-h-96">
                <pre className="text-xs text-gray-100 whitespace-pre-wrap">
                  {JSON.stringify(displayedAggregate, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-gray-500">No aggregate data available</p>
          </CardContent>
        </Card>
      )}

      {/* Save Version Dialog */}
      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save Current Version</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Version Description (Optional)
            </label>
            <input
              type="text"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
              placeholder="E.g., After milestone updates"
              value={saveDescription}
              onChange={(e) => setSaveDescription(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSaveDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveVersion}>
              Save Version
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Revert Confirmation Dialog */}
      <Dialog open={showRevertDialog} onOpenChange={setShowRevertDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revert to Previous Version</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <p className="text-sm text-gray-700">
              Are you sure you want to revert to this version? This will:
            </p>
            <ul className="list-disc list-inside text-sm text-gray-700 mt-2 space-y-1">
              <li>Save the current aggregate as a new version (backup)</li>
              <li>Replace the current aggregate with the selected version</li>
              <li>This action can be undone by reverting to another version</li>
            </ul>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowRevertDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleRevert}>
              Confirm Revert
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
