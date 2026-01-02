import { makeAuthenticatedRequest } from './auth';

const API_BASE = import.meta.env.VITE_API_BASE;

export interface Project {
  id: number;
  name: string;
  description: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  created_by_id?: number;
  ai_generated?: boolean;
  ai_aggregate_response?: any;
}

export interface ProjectStatistics {
  test_cases_count: number;
  test_runs_count: number;
  total_milestones: number;
  completed_milestones: number;
  health_percentage: number;
  total_executions: number;
  passed_executions: number;
  failed_executions: number;
  pending_executions: number;
}

// NEW: Lightweight interface for fast list view
export interface ProjectSummary {
  id: number;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;  // Add missing updated_at field
  test_cases_count: number;
  test_runs_count: number;
  health_percentage: number;  // Add real health percentage
}

export interface ProjectWithStats extends Project {
  statistics: ProjectStatistics;
}

export interface CreateProjectData {
  name: string;
  description: string;
  participant_emails?: string[];
  ai_generated?: boolean;
  ai_aggregate_response?: any;
}

export interface UpdateProjectData {
  name?: string;
  description?: string;
}

export interface ProjectMember {
  id: number;
  project_id: number;
  user_id: number;
  user_email: string;
  user_name: string;
  role: string;
  added_at: string;
  added_by_id: number;
}

export class ProjectService {
  async getProjects(): Promise<Project[]> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch projects');
    }

    return response.json();
  }

  // NEW: Fast endpoint for project list view
  async getProjectsSummary(): Promise<ProjectSummary[]> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/summary`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch projects summary');
    }

    return response.json();
  }

  // NEW: Get detailed stats for specific project
  async getProjectStats(projectId: number): Promise<ProjectWithStats> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}/stats`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch project statistics');
    }

    return response.json();
  }

  // NEW: Fast overview summary endpoint for project overview tab
  async getProjectOverviewSummary(projectId: number): Promise<{
    project_id: number;
    project: {
      id: number;
      name: string;
      description: string;
      created_at: string;
      updated_at: string;
      ai_generated?: boolean;
      ai_aggregate_response?: any;
    };
    test_stats: {
      total_test_cases: number;
      passed: number;
      failed: number;
      pending: number;
      pass_rate: number;
    };
    recent_test_runs: Array<{
      id: number;
      name: string;
      created_at: string;
      status: string;
    }>;
  }> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}/overview-summary`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch project overview summary');
    }

    return response.json();
  }

  // NEW: Get test suites with counts (optimized)
  async getProjectSuitesWithCounts(projectId: number): Promise<Array<{
    id: number;
    name: string;
    description: string;
    created_at: string;
    test_case_count: number;
  }>> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}/suites-with-counts`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch project suites with counts');
    }

    return response.json();
  }

  // LEGACY: Kept for backwards compatibility - will be slower
  async getProjectsWithStats(): Promise<ProjectWithStats[]> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/with-stats`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch projects with statistics');
    }

    return response.json();
  }

  async getProject(projectId: number): Promise<Project> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch project');
    }

    return response.json();
  }

  async createProject(projectData: CreateProjectData): Promise<Project> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/`, {
      method: 'POST',
      body: JSON.stringify(projectData)
    });

    if (!response.ok) {
      const errorData = await response.text();
      let errorMessage;
      try {
        const errorJson = JSON.parse(errorData);
        if (Array.isArray(errorJson)) {
          // Handle validation errors
          errorMessage = errorJson.map(err => err.msg).join(', ');
        } else if (errorJson.detail) {
          errorMessage = errorJson.detail;
        } else {
          errorMessage = JSON.stringify(errorJson);
        }
      } catch {
        errorMessage = errorData || 'Failed to create project';
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async deleteProject(projectId: number): Promise<void> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to delete project');
    }
  }

  async renameProject(projectId: number, newName: string): Promise<Project> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}/rename?new_name=${encodeURIComponent(newName)}`, {
      method: 'PATCH'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to rename project');
    }

    return response.json();
  }

  async updateProject(projectId: number, projectData: UpdateProjectData): Promise<Project> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}`, {
      method: 'PUT',
      body: JSON.stringify(projectData)
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to update project');
    }

    return response.json();
  }

  async getProjectMembers(projectId: number): Promise<ProjectMember[]> {
    const response = await makeAuthenticatedRequest(`${API_BASE}/api/v1/projects/${projectId}/members`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(errorData || 'Failed to fetch project members');
    }

    return response.json();
  }
}

export const projectService = new ProjectService();