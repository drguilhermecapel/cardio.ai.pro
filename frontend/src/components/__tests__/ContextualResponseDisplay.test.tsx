import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import ContextualResponseDisplay from '../ContextualResponseDisplay'

const theme = createTheme()

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ThemeProvider theme={theme}>
    {children}
  </ThemeProvider>
)

describe('ContextualResponseDisplay', () => {
  const mockOnTryAgain = vi.fn()
  const mockOnLearnMore = vi.fn()
  const mockOnGetHelp = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  const defaultProps = {
    response: {
      message: 'Test message',
      explanation: 'Test explanation',
      tips: ['Tip 1', 'Tip 2'],
      visual_guide: 'Visual guide content',
    },
    category: 'medical_document',
    confidence: 0.8,
    onTryAgain: mockOnTryAgain,
    onLearnMore: mockOnLearnMore,
    onGetHelp: mockOnGetHelp,
  }

  describe('Basic Rendering', () => {
    it('renders the main message correctly', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.getByText('Test message')).toBeInTheDocument()
    })

    it('displays category and confidence information', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.getByText('medical document detected')).toBeInTheDocument()
      expect(screen.getByText('80.0% confidence')).toBeInTheDocument()
    })

    it('shows explanation when provided', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.getByText('Test explanation')).toBeInTheDocument()
    })

    it('renders tips section when tips are provided', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.getByText('💡 Helpful Tips')).toBeInTheDocument()
      expect(screen.getByText('• Tip 1')).toBeInTheDocument()
      expect(screen.getByText('• Tip 2')).toBeInTheDocument()
    })

    it('displays visual guide when provided', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.getByText('📋 What an ECG looks like')).toBeInTheDocument()
      expect(screen.getByText('Visual guide content')).toBeInTheDocument()
    })
  })

  describe('Category-Specific Behavior', () => {
    it('displays correct chip color for medical document category', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} category="medical_document" />
        </TestWrapper>
      )

      const chip = screen.getByText('medical document detected')
      expect(chip).toBeInTheDocument()
    })

    it('displays correct chip color for X-ray category', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} category="x_ray" />
        </TestWrapper>
      )

      const chip = screen.getByText('x ray detected')
      expect(chip).toBeInTheDocument()
    })

    it('displays correct chip color for food category', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} category="food" />
        </TestWrapper>
      )

      const chip = screen.getByText('food detected')
      expect(chip).toBeInTheDocument()
    })

    it('handles unknown category gracefully', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} category="unknown_category" />
        </TestWrapper>
      )

      const chip = screen.getByText('unknown category detected')
      expect(chip).toBeInTheDocument()
    })
  })

  describe('Humor Response', () => {
    it('displays humor response when provided', () => {
      const propsWithHumor = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          humor_response: 'This is a funny response!',
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithHumor} />
        </TestWrapper>
      )

      expect(screen.getByText('This is a funny response!')).toBeInTheDocument()
    })

    it('does not show humor section when humor response is not provided', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.queryByText(/funny/)).not.toBeInTheDocument()
    })
  })

  describe('Educational Content', () => {
    it('displays educational content in accordion when provided', () => {
      const propsWithEducation = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          educational_content: {
            title: 'Learn About ECGs',
            description: 'ECGs measure heart activity',
            key_features: ['P wave', 'QRS complex', 'T wave'],
            examples: ['Normal sinus rhythm', 'Atrial fibrillation'],
          },
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithEducation} />
        </TestWrapper>
      )

      expect(screen.getByText('📚 Learn About ECGs')).toBeInTheDocument()
    })

    it('expands educational content when accordion is clicked', async () => {
      const propsWithEducation = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          educational_content: {
            title: 'Learn About ECGs',
            description: 'ECGs measure heart activity',
            key_features: ['P wave', 'QRS complex'],
          },
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithEducation} />
        </TestWrapper>
      )

      const accordion = screen.getByText('📚 Learn About ECGs')
      fireEvent.click(accordion)

      await waitFor(() => {
        expect(screen.getByText('ECGs measure heart activity')).toBeInTheDocument()
        expect(screen.getByText('P wave')).toBeInTheDocument()
        expect(screen.getByText('QRS complex')).toBeInTheDocument()
      })
    })
  })

  describe('Adaptive Suggestions', () => {
    it('displays adaptive suggestions when provided', () => {
      const propsWithSuggestions = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          adaptive_suggestions: [
            'Based on your history, try better lighting',
            'Consider using the camera instead of uploading',
          ],
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithSuggestions} />
        </TestWrapper>
      )

      expect(screen.getByText('🎯 Personalized Suggestions')).toBeInTheDocument()
      expect(screen.getByText('• Based on your history, try better lighting')).toBeInTheDocument()
      expect(screen.getByText('• Consider using the camera instead of uploading')).toBeInTheDocument()
    })
  })

  describe('Helpful Actions', () => {
    it('displays helpful actions when provided', () => {
      const propsWithActions = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          helpful_actions: [
            'Take a new photo with better lighting',
            'Use the camera feature instead',
            'Contact support for help',
          ],
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithActions} />
        </TestWrapper>
      )

      expect(screen.getByText('🚀 What you can do next')).toBeInTheDocument()
      expect(screen.getByText('• Take a new photo with better lighting')).toBeInTheDocument()
      expect(screen.getByText('• Use the camera feature instead')).toBeInTheDocument()
      expect(screen.getByText('• Contact support for help')).toBeInTheDocument()
    })
  })

  describe('Action Buttons', () => {
    it('renders Try Again button and calls onTryAgain when clicked', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      const tryAgainButton = screen.getByText('📷 Try Again')
      expect(tryAgainButton).toBeInTheDocument()

      fireEvent.click(tryAgainButton)
      expect(mockOnTryAgain).toHaveBeenCalledTimes(1)
    })

    it('renders Learn More button when educational content is provided', () => {
      const propsWithEducation = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          educational_content: {
            title: 'Learn About ECGs',
            description: 'Educational content',
          },
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithEducation} />
        </TestWrapper>
      )

      const learnMoreButton = screen.getByText('📚 Learn More')
      expect(learnMoreButton).toBeInTheDocument()

      fireEvent.click(learnMoreButton)
      expect(mockOnLearnMore).toHaveBeenCalledTimes(1)
    })

    it('does not render Learn More button when educational content is not provided', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(screen.queryByText('📚 Learn More')).not.toBeInTheDocument()
    })

    it('renders Get Help button and calls onGetHelp when clicked', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      const getHelpButton = screen.getByText('💬 Get Help')
      expect(getHelpButton).toBeInTheDocument()

      fireEvent.click(getHelpButton)
      expect(mockOnGetHelp).toHaveBeenCalledTimes(1)
    })
  })

  describe('Privacy Notice', () => {
    it('always displays privacy notice', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      expect(
        screen.getByText('🔒 Privacy Notice: Your image was not stored since it doesn\'t contain ECG data.')
      ).toBeInTheDocument()
    })
  })

  describe('Confidence Levels', () => {
    it('displays high confidence correctly', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} confidence={0.95} />
        </TestWrapper>
      )

      expect(screen.getByText('95.0% confidence')).toBeInTheDocument()
    })

    it('displays low confidence correctly', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} confidence={0.25} />
        </TestWrapper>
      )

      expect(screen.getByText('25.0% confidence')).toBeInTheDocument()
    })

    it('handles zero confidence', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} confidence={0.0} />
        </TestWrapper>
      )

      expect(screen.getByText('0.0% confidence')).toBeInTheDocument()
    })
  })

  describe('Optional Props', () => {
    it('handles missing explanation gracefully', () => {
      const propsWithoutExplanation = {
        ...defaultProps,
        response: {
          message: 'Test message',
          tips: ['Tip 1'],
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithoutExplanation} />
        </TestWrapper>
      )

      expect(screen.getByText('Test message')).toBeInTheDocument()
      expect(screen.queryByText('Test explanation')).not.toBeInTheDocument()
    })

    it('handles missing tips gracefully', () => {
      const propsWithoutTips = {
        ...defaultProps,
        response: {
          message: 'Test message',
          explanation: 'Test explanation',
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithoutTips} />
        </TestWrapper>
      )

      expect(screen.getByText('Test message')).toBeInTheDocument()
      expect(screen.queryByText('💡 Helpful Tips')).not.toBeInTheDocument()
    })

    it('handles missing visual guide gracefully', () => {
      const propsWithoutVisualGuide = {
        ...defaultProps,
        response: {
          message: 'Test message',
          tips: ['Tip 1'],
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithoutVisualGuide} />
        </TestWrapper>
      )

      expect(screen.getByText('Test message')).toBeInTheDocument()
      expect(screen.queryByText('📋 What an ECG looks like')).not.toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('has proper ARIA labels for buttons', () => {
      render(
        <TestWrapper>
          <ContextualResponseDisplay {...defaultProps} />
        </TestWrapper>
      )

      const tryAgainButton = screen.getByRole('button', { name: /try again/i })
      const getHelpButton = screen.getByRole('button', { name: /get help/i })

      expect(tryAgainButton).toBeInTheDocument()
      expect(getHelpButton).toBeInTheDocument()
    })

    it('has proper heading structure', () => {
      const propsWithEducation = {
        ...defaultProps,
        response: {
          ...defaultProps.response,
          educational_content: {
            title: 'Learn About ECGs',
            description: 'Educational content',
          },
        },
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithEducation} />
        </TestWrapper>
      )

      const headings = screen.getAllByRole('heading', { level: 6 })
      expect(headings.length).toBeGreaterThan(0)
    })
  })

  describe('Error Handling', () => {
    it('handles empty response object gracefully', () => {
      const propsWithEmptyResponse = {
        ...defaultProps,
        response: {},
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithEmptyResponse} />
        </TestWrapper>
      )

      expect(screen.getByText('📷 Try Again')).toBeInTheDocument()
    })

    it('handles undefined callback functions gracefully', () => {
      const propsWithoutCallbacks = {
        ...defaultProps,
        onTryAgain: undefined,
        onLearnMore: undefined,
        onGetHelp: undefined,
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay {...propsWithoutCallbacks} />
        </TestWrapper>
      )

      expect(screen.getByText('📷 Try Again')).toBeInTheDocument()
      expect(screen.getByText('💬 Get Help')).toBeInTheDocument()
    })
  })

  describe('Integration Scenarios', () => {
    it('renders complete response with all features', () => {
      const completeResponse = {
        message: 'Food image detected',
        explanation: 'This appears to be a food image, not an ECG',
        tips: [
          'ECGs have a grid pattern',
          'ECGs show heart rhythm waves',
          'Try taking a photo of an actual ECG printout',
        ],
        visual_guide: 'An ECG typically shows multiple leads with waveforms on a grid background',
        educational_content: {
          title: 'Understanding ECGs',
          description: 'Electrocardiograms measure electrical activity of the heart',
          key_features: ['P wave', 'QRS complex', 'T wave', 'Grid pattern'],
          examples: ['12-lead ECG', 'Rhythm strip', 'Holter monitor output'],
        },
        helpful_actions: [
          'Take a photo of an ECG printout',
          'Use better lighting',
          'Ensure the entire ECG is visible',
        ],
        humor_response: '🍕 Delicious! But we analyze heartbeats, not recipes!',
        adaptive_suggestions: [
          'Based on your previous attempts, try using the camera feature',
          'Consider asking a healthcare provider for an ECG printout',
        ],
      }

      render(
        <TestWrapper>
          <ContextualResponseDisplay
            response={completeResponse}
            category="food"
            confidence={0.92}
            onTryAgain={mockOnTryAgain}
            onLearnMore={mockOnLearnMore}
            onGetHelp={mockOnGetHelp}
          />
        </TestWrapper>
      )

      expect(screen.getByText('Food image detected')).toBeInTheDocument()
      expect(screen.getByText('🍕 Delicious! But we analyze heartbeats, not recipes!')).toBeInTheDocument()
      expect(screen.getByText('💡 Helpful Tips')).toBeInTheDocument()
      expect(screen.getByText('📋 What an ECG looks like')).toBeInTheDocument()
      expect(screen.getByText('📚 Understanding ECGs')).toBeInTheDocument()
      expect(screen.getByText('🎯 Personalized Suggestions')).toBeInTheDocument()
      expect(screen.getByText('🚀 What you can do next')).toBeInTheDocument()
      expect(screen.getByText('92.0% confidence')).toBeInTheDocument()
    })
  })
})
