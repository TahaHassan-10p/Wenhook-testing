import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Loader2 } from 'lucide-react';
import { testPlanService, type CreateTestPlanData, type UpdateTestPlanData, type TestPlan } from '../../services/test-plans';

interface TestPlanModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTestPlanUpdated?: () => void;
  projectId: number;
  testPlan?: TestPlan | null; // null for create, testPlan object for edit
}

export function TestPlanModal({ isOpen, onClose, onTestPlanUpdated, projectId, testPlan }: TestPlanModalProps) {
  const [name, setName] = useState(testPlan?.name || '');
  const [description, setDescription] = useState(testPlan?.description || '');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isEditing = !!testPlan;

  // Update form when testPlan prop changes
  useEffect(() => {
    if (testPlan) {
      setName(testPlan.name);
      setDescription(testPlan.description || '');
    } else {
      setName('');
      setDescription('');
    }
  }, [testPlan]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Test plan name is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      if (isEditing && testPlan) {
        // Update existing test plan
        const updateData: UpdateTestPlanData = {
          name: name.trim(),
          description: description.trim() || undefined
        };
        
        await testPlanService.updateTestPlan(testPlan.id, updateData);
        // Show success toast
        const event = new CustomEvent('toast', {
          detail: { type: 'success', message: `Test plan "${name}" updated successfully!` }
        });
        window.dispatchEvent(event);
      } else {
        // Create new test plan
        const createData: CreateTestPlanData = {
          name: name.trim(),
          description: description.trim() || undefined,
          project_id: projectId
        };
        
        await testPlanService.createTestPlan(createData);
        // Show success toast
        const event = new CustomEvent('toast', {
          detail: { type: 'success', message: `Test plan "${name}" created successfully!` }
        });
        window.dispatchEvent(event);
      }
      
      onTestPlanUpdated?.();
      handleClose();
    } catch (error) {
      console.error('Failed to save test plan:', error);
      setError('Failed to save test plan. Please try again.');
      // Show error toast
      const event = new CustomEvent('toast', {
        detail: { type: 'error', message: 'Failed to save test plan. Please try again.' }
      });
      window.dispatchEvent(event);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    if (!isLoading) {
      setName('');
      setDescription('');
      setError(null);
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isEditing ? 'Edit Test Plan' : 'Create Test Plan'}
          </DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="text-red-600 text-sm bg-red-50 p-2 rounded">
              {error}
            </div>
          )}
          
          <div className="space-y-2">
            <Label htmlFor="name">Name *</Label>
            <Input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter test plan name..."
              required
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter test plan description..."
              rows={3}
            />
          </div>
          
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading || !name.trim()}>
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isEditing ? 'Update' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
